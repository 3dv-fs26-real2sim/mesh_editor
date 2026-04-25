"""Image library: thumbnail grid + enlarged view with eyedropper."""
from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QIcon, QImage, QPainter, QPixmap
from PyQt5.QtWidgets import (
    QFileDialog,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

THUMB = 96
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class EnlargedView(QGraphicsView):
    color_picked = pyqtSignal(int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)
        self.setRenderHint(QPainter.SmoothPixmapTransform, True)
        self._item = QGraphicsPixmapItem()
        self.scene_obj.addItem(self._item)
        self._image: QImage | None = None
        self.setMouseTracking(True)
        self.setCursor(Qt.CrossCursor)

    def set_image(self, path: Path) -> None:
        img = QImage(str(path))
        if img.isNull():
            return
        self._image = img.convertToFormat(QImage.Format_RGB32)
        self._item.setPixmap(QPixmap.fromImage(self._image))
        self.scene_obj.setSceneRect(self._item.boundingRect())
        self.fitInView(self._item, Qt.KeepAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._image is not None:
            self.fitInView(self._item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        factor = 1.2 if delta > 0 else 0.83
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self._image is not None:
            sp: QPointF = self.mapToScene(event.pos())
            x = int(sp.x())
            y = int(sp.y())
            if 0 <= x < self._image.width() and 0 <= y < self._image.height():
                c = self._image.pixelColor(x, y)
                self.color_picked.emit(c.red(), c.green(), c.blue())
                return
        super().mousePressEvent(event)


class LibraryPanel(QWidget):
    color_picked = pyqtSignal(int, int, int)
    folder_loaded = pyqtSignal(Path)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.btn_import = QPushButton("Import folder...", self)
        self.btn_import.clicked.connect(self._on_import)
        layout.addWidget(self.btn_import)

        splitter = QSplitter(Qt.Vertical, self)
        layout.addWidget(splitter, 1)

        self.list = QListWidget(self)
        self.list.setViewMode(QListView.IconMode)
        self.list.setIconSize(QSize(THUMB, THUMB))
        self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static)
        self.list.setSpacing(4)
        self.list.itemClicked.connect(self._on_item)
        splitter.addWidget(self.list)

        self.enlarged = EnlargedView(self)
        self.enlarged.color_picked.connect(self.color_picked.emit)
        splitter.addWidget(self.enlarged)
        splitter.setSizes([200, 400])

    def _on_import(self):
        d = QFileDialog.getExistingDirectory(self, "Pick image folder")
        if not d:
            return
        self.load_folder(Path(d))

    def load_folder(self, folder: Path):
        self.list.clear()
        for p in sorted(folder.iterdir()):
            if p.suffix.lower() not in IMAGE_EXTS:
                continue
            qimg = QImage(str(p))
            if qimg.isNull():
                continue
            scaled = qimg.scaled(
                THUMB,
                THUMB,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
            item = QListWidgetItem(p.name)
            item.setIcon(QIcon(QPixmap.fromImage(scaled)))
            item.setData(Qt.UserRole, str(p))
            item.setToolTip(str(p))
            self.list.addItem(item)
        self.folder_loaded.emit(folder)

    def _on_item(self, item: QListWidgetItem):
        path = Path(item.data(Qt.UserRole))
        self.enlarged.set_image(path)
