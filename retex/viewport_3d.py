"""3D viewport: pyvistaqt QtInteractor showing the textured mesh, with a
paint mode that converts left-mouse drags into UV-space brush strokes.

We use a Qt event filter on the QtInteractor widget instead of VTK observers.
VTK observers proved flaky for release events when AbortFlagOn() consumed the
press, so paint sometimes "stuck on" after the user released the mouse.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv
import vtk
from PyQt5.QtCore import QEvent, QObject, QPoint, QTimer, Qt
from PyQt5.QtWidgets import QFrame, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor

from .texture_state import TextureState
from .uv_mapper import UVMapper


class Viewport3D(QWidget):
    def __init__(
        self,
        mesh: pv.PolyData,
        state: TextureState,
        uv_mapper: UVMapper,
        parent=None,
    ):
        super().__init__(parent)
        self.mesh = mesh
        self.state = state
        self.uv_mapper = uv_mapper

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.frame = QFrame(self)
        layout.addWidget(self.frame)
        frame_layout = QVBoxLayout(self.frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)

        self.plotter = QtInteractor(self.frame)
        frame_layout.addWidget(self.plotter)

        self.plotter.set_background("black")
        self._texture = pv.numpy_to_texture(state.buffer)
        kwargs = dict(reset_camera=True, smooth_shading=True)
        if uv_mapper.has_uv:
            kwargs["texture"] = self._texture
        else:
            kwargs["color"] = "lightgray"
        self.actor = self.plotter.add_mesh(mesh, **kwargs)
        self.plotter.add_axes()

        self.brush_radius = 16
        self.brush_color: tuple[int, int, int] = (255, 0, 0)

        self._paint_mode = False
        self._painting = False
        self._last_uv: tuple[int, int] | None = None
        self._cell_picker = vtk.vtkCellPicker()
        self._cell_picker.SetTolerance(0.0005)

        # Install Qt event filter on the QtInteractor widget so we can intercept
        # mouse events before VTK's interactor style sees them.
        self.plotter.installEventFilter(self)

        # Subscribe to texture buffer changes to push them back to VTK.
        self.state.changed.connect(self._on_texture_changed)

        # Throttle 3D re-renders during fast drags. The texture rebuild is the
        # expensive part (full-buffer deep copy + GPU re-upload), so we coalesce
        # it into the render timer rather than running it per brush stamp —
        # otherwise fast strokes block the GUI thread and Qt drops mouse moves,
        # which is what produced "dots instead of strokes".
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self._do_render)
        self._render_pending = False
        self._texture_dirty = False

    def set_paint_mode(self, on: bool) -> None:
        self._paint_mode = on
        if on:
            self.plotter.setCursor(Qt.CrossCursor)
        else:
            self.plotter.setCursor(Qt.ArrowCursor)
            # If the user toggles out mid-stroke, end it cleanly.
            if self._painting:
                self.state.end_stroke()
                self._painting = False
                self._last_uv = None

    # ---- Qt event filtering ----

    def eventFilter(self, obj: QObject, event) -> bool:
        if not self._paint_mode or obj is not self.plotter:
            return super().eventFilter(obj, event)
        et = event.type()
        if et == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
            self._paint_at_qt_pos(event.pos(), starting_stroke=True)
            self._painting = True
            return True
        if et == QEvent.MouseMove and self._painting:
            self._paint_at_qt_pos(event.pos(), starting_stroke=False)
            return True
        if et == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
            if self._painting:
                self.state.end_stroke()
                self._painting = False
                self._last_uv = None
            return True
        # Right/middle clicks etc. fall through to VTK for camera control.
        return super().eventFilter(obj, event)

    # ---- painting ----

    def _qt_to_vtk_xy(self, pos: QPoint) -> tuple[int, int]:
        """Qt widget coords (origin top-left) -> VTK display coords (origin bottom-left)."""
        h = self.plotter.height()
        return int(pos.x()), int(h - pos.y())

    def _paint_at_qt_pos(self, qt_pos: QPoint, starting_stroke: bool) -> None:
        x, y = self._qt_to_vtk_xy(qt_pos)
        renderer = self.plotter.renderer
        if not self._cell_picker.Pick(x, y, 0, renderer):
            # Picking missed; don't break the stroke - we'll resume on next hit.
            return
        cell_id = self._cell_picker.GetCellId()
        if cell_id < 0:
            return
        world_xyz = np.asarray(self._cell_picker.GetPickPosition(), dtype=float)
        uv_px = self.uv_mapper.world_to_uv_px(
            cell_id, world_xyz, self.state.width, self.state.height
        )
        if uv_px is None:
            return

        if starting_stroke:
            self.state.begin_stroke()
            self.state.paint_disc(
                uv_px[0], uv_px[1], self.brush_radius, self.brush_color
            )
            self._last_uv = uv_px
            return

        last = self._last_uv
        if last is None:
            self.state.paint_disc(
                uv_px[0], uv_px[1], self.brush_radius, self.brush_color
            )
            self._last_uv = uv_px
            return

        # Heuristic: if the new sample is wildly far (e.g. picker jumped to a
        # different UV island across a seam), don't draw a long line through
        # unrelated texture regions.
        dist = float(np.hypot(uv_px[0] - last[0], uv_px[1] - last[1]))
        max_continuous = max(8.0 * self.brush_radius, 200.0)
        if dist > max_continuous:
            self.state.paint_disc(
                uv_px[0], uv_px[1], self.brush_radius, self.brush_color
            )
        else:
            self.state.paint_segment(
                last[0], last[1], uv_px[0], uv_px[1],
                self.brush_radius, self.brush_color,
            )
        self._last_uv = uv_px

    # ---- texture sync ----

    def _on_texture_changed(self, _rect) -> None:
        # Mark the texture dirty and let the throttled render coalesce the
        # rebuild. Doing the rebuild here (per stamp) starves the Qt event loop.
        self._texture_dirty = True
        self._schedule_render()

    def _schedule_render(self) -> None:
        if self._render_pending:
            return
        self._render_pending = True
        self._render_timer.start(16)  # ~60Hz cap

    def _do_render(self) -> None:
        self._render_pending = False
        if self._texture_dirty:
            new_tex = pv.numpy_to_texture(self.state.buffer)
            self.actor.SetTexture(new_tex)
            self._texture = new_tex
            self._texture_dirty = False
        self.plotter.render()

    def close(self) -> None:
        self.plotter.removeEventFilter(self)
        self.plotter.close()
        super().close()
