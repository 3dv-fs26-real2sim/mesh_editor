"""Mesh retexturing tool.

Open a textured OBJ, paint the texture by sampling colors from a library of
reference images. Saves a new asset folder leaving the original untouched.

Usage:
    python retexture.py objects/ball
    python retexture.py objects/ball --output objects
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QIcon, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QColorDialog,
    QDockWidget,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QStatusBar,
    QToolBar,
    QWidget,
)

from retex.io_utils import LoadedMesh
from retex.library_panel import LibraryPanel
from retex.texture_state import TextureState
from retex.uv_mapper import UVMapper
from retex.viewport_3d import Viewport3D
from retex.viewport_uv import UVViewport

DEFAULT_BRUSH = 16
MIN_BRUSH = 1
MAX_BRUSH = 256


class RetextureWindow(QMainWindow):
    def __init__(self, source_dir: Path, output_root: Path):
        super().__init__()
        self.source_dir = source_dir
        self.output_root = output_root
        self.setWindowTitle(f"Mesh Retexture — {source_dir.name}")
        self.resize(1600, 980)

        self.loaded = LoadedMesh(source_dir)
        if self.loaded.tex_path is None:
            QMessageBox.warning(
                self,
                "No texture found",
                "No texture image was found in the source folder. The retexture pane will start with a blank gray buffer.",
            )
        self.state = TextureState(self.loaded.tex_path)
        self.uv_mapper = UVMapper(self.loaded.mesh)

        self.brush_color: tuple[int, int, int] = (255, 0, 0)
        self.brush_radius = DEFAULT_BRUSH

        self._build_central()
        self._build_library_dock()
        self._build_toolbar()
        self._build_status_bar()
        self._wire_brush_state()

        # Wireframe overlay for the UV pane.
        self.uv_view.set_uv_wireframe(
            self.uv_mapper.triangle_uv_edges_px(self.state.width, self.state.height)
        )

    # ---- UI assembly ----

    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal, self)
        self.viewport_3d = Viewport3D(
            self.loaded.mesh, self.state, self.uv_mapper, self
        )
        self.uv_view = UVViewport(self.state, self)
        splitter.addWidget(self.viewport_3d)
        splitter.addWidget(self.uv_view)
        splitter.setSizes([900, 700])
        self.setCentralWidget(splitter)
        self.uv_view.on_hover_uv = self._on_uv_hover

    def _build_library_dock(self):
        self.library = LibraryPanel(self)
        dock = QDockWidget("Reference images", self)
        dock.setWidget(self.library)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.library.color_picked.connect(self._on_color_picked)

    def _build_toolbar(self):
        tb = QToolBar("Tools", self)
        tb.setMovable(False)
        self.addToolBar(tb)

        self.act_paint = QAction("Paint (B)", self, checkable=True)
        self.act_paint.setShortcut("B")
        self.act_paint.toggled.connect(self._on_paint_toggle)
        tb.addAction(self.act_paint)

        self.act_pan = QAction("Camera (Space)", self, checkable=True)
        self.act_pan.setShortcut(Qt.Key_Space)
        self.act_pan.setChecked(True)
        self.act_pan.toggled.connect(self._on_pan_toggle)
        tb.addAction(self.act_pan)

        tb.addSeparator()

        # Color swatch
        self.color_btn = QPushButton(self)
        self.color_btn.setFixedSize(32, 32)
        self._refresh_swatch()
        self.color_btn.clicked.connect(self._on_pick_color_dialog)
        tb.addWidget(QLabel(" Color: ", self))
        tb.addWidget(self.color_btn)

        tb.addSeparator()

        # Brush size
        tb.addWidget(QLabel(" Brush: ", self))
        self.brush_slider = QSlider(Qt.Horizontal, self)
        self.brush_slider.setRange(MIN_BRUSH, MAX_BRUSH)
        self.brush_slider.setValue(DEFAULT_BRUSH)
        self.brush_slider.setFixedWidth(220)
        self.brush_slider.valueChanged.connect(self._on_brush_size)
        tb.addWidget(self.brush_slider)
        self.brush_label = QLabel(f"{DEFAULT_BRUSH}px", self)
        self.brush_label.setMinimumWidth(48)
        tb.addWidget(self.brush_label)

        tb.addSeparator()

        # UV wireframe toggle
        self.act_wire = QAction("UV wireframe", self, checkable=True)
        self.act_wire.setChecked(False)
        self.act_wire.toggled.connect(self.uv_view.set_wireframe_visible)
        tb.addAction(self.act_wire)

        tb.addSeparator()

        # Undo / Redo
        self.act_undo = QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.state.undo)
        tb.addAction(self.act_undo)

        self.act_redo = QAction("Redo", self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(self.state.redo)
        tb.addAction(self.act_redo)

        tb.addSeparator()

        self.act_save = QAction("Save", self)
        self.act_save.setShortcut(QKeySequence.Save)
        self.act_save.triggered.connect(self._on_save)
        tb.addAction(self.act_save)

        # Brush size keys
        for key, delta in (("[", -2), ("]", 2)):
            act = QAction(f"Brush {key}", self)
            act.setShortcut(key)
            act.triggered.connect(lambda _=False, d=delta: self._adjust_brush(d))
            self.addAction(act)

    def _build_status_bar(self):
        sb = QStatusBar(self)
        self.setStatusBar(sb)
        self.status_color = QLabel("rgb(255,0,0)", self)
        self.status_brush = QLabel("brush 16px", self)
        self.status_uv = QLabel("uv: -", self)
        sb.addPermanentWidget(self.status_color)
        sb.addPermanentWidget(self.status_brush)
        sb.addPermanentWidget(self.status_uv)

    def _wire_brush_state(self):
        self._on_paint_toggle(False)  # start in camera mode
        self._sync_brush_to_viewports()

    # ---- brush wiring ----

    def _on_paint_toggle(self, on: bool):
        if on and self.act_pan.isChecked():
            self.act_pan.setChecked(False)
        self.viewport_3d.set_paint_mode(on)

    def _on_pan_toggle(self, on: bool):
        if on and self.act_paint.isChecked():
            self.act_paint.setChecked(False)
            self.viewport_3d.set_paint_mode(False)

    def _on_pick_color_dialog(self):
        c = QColorDialog.getColor(QColor(*self.brush_color), self, "Pick color")
        if c.isValid():
            self._set_color(c.red(), c.green(), c.blue())

    def _on_color_picked(self, r: int, g: int, b: int):
        self._set_color(r, g, b)

    def _set_color(self, r: int, g: int, b: int):
        self.brush_color = (r, g, b)
        self._refresh_swatch()
        self.status_color.setText(f"rgb({r},{g},{b})")
        self._sync_brush_to_viewports()

    def _refresh_swatch(self):
        pm = QPixmap(28, 28)
        pm.fill(QColor(*self.brush_color))
        self.color_btn.setIcon(QIcon(pm))

    def _on_brush_size(self, v: int):
        self.brush_radius = int(v)
        self.brush_label.setText(f"{v}px")
        self.status_brush.setText(f"brush {v}px")
        self._sync_brush_to_viewports()

    def _adjust_brush(self, delta: int):
        v = max(MIN_BRUSH, min(MAX_BRUSH, self.brush_slider.value() + delta))
        self.brush_slider.setValue(v)

    def _sync_brush_to_viewports(self):
        self.uv_view.brush_radius = self.brush_radius
        self.uv_view.brush_color = self.brush_color
        self.viewport_3d.brush_radius = self.brush_radius
        self.viewport_3d.brush_color = self.brush_color

    def _on_uv_hover(self, px: int, py: int):
        self.status_uv.setText(f"uv: px=({px},{py})")

    # ---- save ----

    def _on_save(self):
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_root / f"{self.loaded.name}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Save texture as PNG.
        new_tex_name = f"{self.loaded.name}.png"
        self.state.save_png(out_dir / new_tex_name)
        # Copy OBJ unchanged.
        shutil.copy2(self.loaded.obj_path, out_dir / self.loaded.obj_path.name)
        # Rewrite MTL: replace map_Kd with new texture filename.
        if self.loaded.mtl_path.exists():
            self._rewrite_mtl(
                self.loaded.mtl_path,
                out_dir / self.loaded.mtl_path.name,
                new_tex_name,
            )
        # Provenance.
        meta = {
            "kind": "retexture",
            "timestamp": ts,
            "source_dir": str(self.source_dir),
            "source_obj": str(self.loaded.obj_path),
            "source_texture": str(self.loaded.tex_path) if self.loaded.tex_path else None,
            "new_texture": new_tex_name,
            "texture_size": [self.state.width, self.state.height],
        }
        (out_dir / "retex.json").write_text(json.dumps(meta, indent=2))
        self.statusBar().showMessage(f"Saved -> {out_dir}", 5000)
        QMessageBox.information(self, "Saved", f"Retextured mesh saved to:\n{out_dir}")

    @staticmethod
    def _rewrite_mtl(src: Path, dst: Path, new_texture: str) -> None:
        """Copy MTL line-by-line, replacing every map_Kd <whatever> with the new file."""
        out_lines: list[str] = []
        for raw in src.read_text().splitlines():
            stripped = raw.strip()
            if stripped.lower().startswith("map_kd"):
                # Preserve leading whitespace if any.
                lead = raw[: len(raw) - len(raw.lstrip())]
                out_lines.append(f"{lead}map_Kd {new_texture}")
            else:
                out_lines.append(raw)
        dst.write_text("\n".join(out_lines) + "\n")

    def closeEvent(self, event):
        try:
            self.viewport_3d.close()
        except Exception:
            pass
        super().closeEvent(event)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="Folder containing <name>.obj")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output root (default: parent of <source>)",
    )
    args = ap.parse_args()
    src = args.source.resolve()
    if not src.is_dir():
        sys.exit(f"Not a directory: {src}")
    out_root = args.output.resolve() if args.output else src.parent

    app = QApplication(sys.argv)
    win = RetextureWindow(src, out_root)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
