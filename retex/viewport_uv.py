"""UV / texture pane: paintable QGraphicsView showing the texture buffer
with an optional UV wireframe overlay.
"""
from __future__ import annotations

import numpy as np
from PyQt5.QtCore import QPointF, QRect, QRectF, Qt
from PyQt5.QtGui import (
    QColor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PyQt5.QtWidgets import (
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from .texture_state import TextureState


def _numpy_to_qimage(arr: np.ndarray) -> QImage:
    h, w, _ = arr.shape
    # PyQt5 QImage shares memory with the buffer; we copy() to detach.
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class UVViewport(QGraphicsView):
    def __init__(self, state: TextureState, parent=None):
        super().__init__(parent)
        self.state = state
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QPainter.Antialiasing, False)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setMouseTracking(True)

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setShapeMode(QGraphicsPixmapItem.BoundingRectShape)
        self.scene_obj.addItem(self._pixmap_item)

        self._wire_item: QGraphicsPathItem | None = None
        self._wire_visible = False

        # Brush state plumbed in from MainWindow.
        self.brush_radius = 16
        self.brush_color: tuple[int, int, int] = (255, 0, 0)
        self.on_hover_uv = None  # callback(px, py)

        self._painting = False
        self._panning = False
        self._pan_last: QPointF | None = None
        self._last_uv: tuple[int, int] | None = None
        self._refresh_pixmap_full()
        self.state.changed.connect(self._on_changed)
        self.scene_obj.setSceneRect(QRectF(0, 0, self.state.width, self.state.height))
        self.fitInView(self.scene_obj.sceneRect(), Qt.KeepAspectRatio)

    def set_uv_wireframe(self, edges_px: np.ndarray | None) -> None:
        if self._wire_item is not None:
            self.scene_obj.removeItem(self._wire_item)
            self._wire_item = None
        if edges_px is None or edges_px.size == 0:
            return
        path = QPainterPath()
        for (a, b) in edges_px:
            path.moveTo(float(a[0]), float(a[1]))
            path.lineTo(float(b[0]), float(b[1]))
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(0, 255, 0, 80))
        pen.setWidthF(0.0)  # cosmetic line - constant pixel width
        pen.setCosmetic(True)
        item.setPen(pen)
        item.setZValue(10)
        item.setVisible(self._wire_visible)
        self.scene_obj.addItem(item)
        self._wire_item = item

    def set_wireframe_visible(self, visible: bool) -> None:
        self._wire_visible = visible
        if self._wire_item is not None:
            self._wire_item.setVisible(visible)

    def _refresh_pixmap_full(self) -> None:
        qimg = _numpy_to_qimage(self.state.buffer)
        self._pixmap_item.setPixmap(QPixmap.fromImage(qimg))

    def _on_changed(self, rect: QRect) -> None:
        # Refresh just the dirty rect for performance on big textures.
        if rect.width() <= 0 or rect.height() <= 0:
            return
        # Build a sub-image from the buffer then paint over the existing pixmap.
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        sub = self.state.buffer[y : y + h, x : x + w]
        qimg = _numpy_to_qimage(np.ascontiguousarray(sub))
        pm = self._pixmap_item.pixmap()
        if pm.isNull():
            self._refresh_pixmap_full()
            return
        painter = QPainter(pm)
        painter.drawImage(x, y, qimg)
        painter.end()
        self._pixmap_item.setPixmap(pm)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)

    def _scene_to_pixel(self, sp: QPointF) -> tuple[int, int] | None:
        x = int(sp.x())
        y = int(sp.y())
        if 0 <= x < self.state.width and 0 <= y < self.state.height:
            return x, y
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_last = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            return
        if event.button() == Qt.LeftButton:
            sp = self.mapToScene(event.pos())
            xy = self._scene_to_pixel(sp)
            if xy is not None:
                self._painting = True
                self.state.begin_stroke()
                self.state.paint_disc(
                    xy[0], xy[1], self.brush_radius, self.brush_color
                )
                self._last_uv = xy
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        sp = self.mapToScene(event.pos())
        xy = self._scene_to_pixel(sp)
        if xy is not None and self.on_hover_uv:
            self.on_hover_uv(xy[0], xy[1])
        if self._panning and self._pan_last is not None:
            delta = event.pos() - self._pan_last
            self._pan_last = event.pos()
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(hbar.value() - delta.x())
            vbar.setValue(vbar.value() - delta.y())
            return
        if self._painting and xy is not None:
            # Interpolate between the last sample and the current one so fast
            # cursor motion produces a continuous stroke instead of dots.
            last = self._last_uv
            if last is None:
                self.state.paint_disc(
                    xy[0], xy[1], self.brush_radius, self.brush_color
                )
            else:
                self.state.paint_segment(
                    last[0], last[1], xy[0], xy[1],
                    self.brush_radius, self.brush_color,
                )
            self._last_uv = xy
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self._pan_last = None
            self.setCursor(Qt.ArrowCursor)
            return
        if event.button() == Qt.LeftButton and self._painting:
            self._painting = False
            self._last_uv = None
            self.state.end_stroke()
            return
        super().mouseReleaseEvent(event)
