"""Central texture buffer + paint primitives.

Single source of truth: a HxWx3 uint8 numpy array. Both the UV pane and the
3D viewport read from it and emit/observe the `changed` signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from PyQt5.QtCore import QObject, QRect, pyqtSignal


class TextureState(QObject):
    changed = pyqtSignal(QRect)  # dirty rect in image (pixel) coords

    def __init__(self, image_path: Path | None):
        super().__init__()
        if image_path is not None and image_path.exists():
            img = Image.open(image_path).convert("RGB")
            self.buffer = np.asarray(img, dtype=np.uint8).copy()
        else:
            self.buffer = np.full((1024, 1024, 3), 200, dtype=np.uint8)
        self._undo: list[tuple[QRect, np.ndarray]] = []
        self._redo: list[tuple[QRect, np.ndarray]] = []
        self._undo_limit = 20
        self._stroke_dirty: QRect | None = None
        self._stroke_snapshot: np.ndarray | None = None

    @property
    def height(self) -> int:
        return self.buffer.shape[0]

    @property
    def width(self) -> int:
        return self.buffer.shape[1]

    def begin_stroke(self) -> None:
        """Snapshot before a stroke so undo can revert the whole stroke at once."""
        self._stroke_dirty = None
        self._stroke_snapshot = self.buffer.copy()

    def end_stroke(self) -> None:
        if self._stroke_dirty is None or self._stroke_snapshot is None:
            self._stroke_dirty = None
            self._stroke_snapshot = None
            return
        r = self._stroke_dirty
        before = self._stroke_snapshot[
            r.y() : r.y() + r.height(), r.x() : r.x() + r.width()
        ].copy()
        self._undo.append((r, before))
        if len(self._undo) > self._undo_limit:
            self._undo.pop(0)
        self._redo.clear()
        self._stroke_dirty = None
        self._stroke_snapshot = None

    def paint_disc(
        self, cx: int, cy: int, radius: int, rgb: tuple[int, int, int]
    ) -> None:
        rect = self._stamp_disc(cx, cy, radius, rgb)
        if rect is not None:
            self._extend_stroke_dirty(rect)
            self.changed.emit(rect)

    def paint_segment(
        self,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        radius: int,
        rgb: tuple[int, int, int],
    ) -> None:
        """Stamp discs along the segment (x0,y0)->(x1,y1) and emit one combined
        `changed` rect. Spacing of radius/2 keeps the line visually solid.
        """
        if radius < 1:
            radius = 1
        dx = x1 - x0
        dy = y1 - y0
        dist = float(np.hypot(dx, dy))
        spacing = max(1.0, radius / 2.0)
        steps = max(1, int(np.ceil(dist / spacing)))
        rgb_arr = np.array(rgb, dtype=np.uint8)
        combined: QRect | None = None
        for i in range(steps + 1):
            t = i / steps if steps else 0.0
            cx = int(round(x0 + t * dx))
            cy = int(round(y0 + t * dy))
            r = self._stamp_disc(cx, cy, radius, rgb_arr)
            if r is None:
                continue
            combined = r if combined is None else combined.united(r)
        if combined is not None:
            self._extend_stroke_dirty(combined)
            self.changed.emit(combined)

    def _stamp_disc(
        self, cx: int, cy: int, radius: int, rgb
    ) -> QRect | None:
        if radius < 1:
            radius = 1
        h, w, _ = self.buffer.shape
        x0 = max(0, cx - radius)
        x1 = min(w, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= radius * radius
        if not mask.any():
            return None
        region = self.buffer[y0:y1, x0:x1]
        if not isinstance(rgb, np.ndarray):
            rgb = np.array(rgb, dtype=np.uint8)
        region[mask] = rgb
        return QRect(x0, y0, x1 - x0, y1 - y0)

    def _extend_stroke_dirty(self, r: QRect) -> None:
        if self._stroke_dirty is None:
            self._stroke_dirty = QRect(r)
        else:
            self._stroke_dirty = self._stroke_dirty.united(r)

    def undo(self) -> None:
        if not self._undo:
            return
        rect, before = self._undo.pop()
        after = self.buffer[
            rect.y() : rect.y() + rect.height(),
            rect.x() : rect.x() + rect.width(),
        ].copy()
        self._redo.append((rect, after))
        self.buffer[
            rect.y() : rect.y() + rect.height(),
            rect.x() : rect.x() + rect.width(),
        ] = before
        self.changed.emit(rect)

    def redo(self) -> None:
        if not self._redo:
            return
        rect, after = self._redo.pop()
        before = self.buffer[
            rect.y() : rect.y() + rect.height(),
            rect.x() : rect.x() + rect.width(),
        ].copy()
        self._undo.append((rect, before))
        self.buffer[
            rect.y() : rect.y() + rect.height(),
            rect.x() : rect.x() + rect.width(),
        ] = after
        self.changed.emit(rect)

    def save_png(self, path: Path) -> None:
        Image.fromarray(self.buffer, mode="RGB").save(path, format="PNG")
