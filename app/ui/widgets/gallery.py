from typing import List, Dict, Tuple, Optional
import numpy as np

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QScrollArea, QGridLayout, QFrame, QLabel, QVBoxLayout


def numpy_rgb_to_qimage(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()

class Tile(QFrame):
    clicked = pyqtSignal(str, int, str)  # h5_path, row_idx, button

    def __init__(self):
        super().__init__()
        self.setFrameStyle(QFrame.Box)
        self.setLineWidth(1)
        self.setStyleSheet("QFrame { background: white; }")

        self.h5_path: str = ""
        self.row_idx: int = -1

        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)

        self.lbl = QLabel()
        self.lbl.setAlignment(Qt.AlignCenter)
        self.lbl.setStyleSheet("font-size: 11px; padding: 2px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)
        lay.addWidget(self.img)
        lay.addWidget(self.lbl)

        self._tile_w = 96
        self._tile_h = 96

    def set_geometry(self, tile_w: int, tile_h: int):
        self._tile_w = int(tile_w)
        self._tile_h = int(tile_h)
        # total tile height includes label line
        self.setFixedSize(self._tile_w + 8, self._tile_h + 26)
        self.img.setFixedSize(self._tile_w, self._tile_h)

    def set_content(self, h5_path: str, row_idx: int, rgb: np.ndarray, label: str, tooltip: str):
        self.h5_path = h5_path
        self.row_idx = int(row_idx)

        qimg = numpy_rgb_to_qimage(rgb)
        pm = QPixmap.fromImage(qimg)
        # ✅ fill the rectangle exactly (no white bars)
        pm2 = pm.scaled(self._tile_w, self._tile_h, Qt.IgnoreAspectRatio, Qt.FastTransformation)

        self.img.setPixmap(pm2)
        self.lbl.setText(label)
        self.setToolTip(tooltip)

    def set_label(self, s: str):
        self.lbl.setText(s)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.h5_path, self.row_idx, "left")
        elif e.button() == Qt.RightButton:
            self.clicked.emit(self.h5_path, self.row_idx, "right")
        super().mousePressEvent(e)

class GalleryWidget(QWidget):
    tile_clicked = pyqtSignal(str, int, str)

    def __init__(self):
        super().__init__()

        self._n_cols = 6
        self._tile_h = 96
        self._tile_w = 96

        self._tiles_data: List[Dict] = []
        self._tile_widgets: List[Tile] = []
        self._key_to_widget: Dict[Tuple[str, int], Tile] = {}

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)

        self.inner = QWidget()
        self.grid = QGridLayout(self.inner)
        self.grid.setContentsMargins(6, 6, 6, 6)
        self.grid.setHorizontalSpacing(6)
        self.grid.setVerticalSpacing(6)

        self.scroll.setWidget(self.inner)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll)

    def set_layout(self, *, n_cols: int, tile_h: int, tile_w: Optional[int] = None):
        self._n_cols = int(n_cols)
        self._tile_h = int(tile_h)
        self._tile_w = int(tile_w) if tile_w is not None else int(tile_h)

        # apply geometry to existing pool
        for t in self._tile_widgets:
            t.set_geometry(self._tile_w, self._tile_h)

        # reflow current tiles
        if self._tiles_data:
            self.set_tiles(self._tiles_data)

    def set_tiles(self, tiles: List[Dict]):
        self._tiles_data = tiles or []
        self._key_to_widget.clear()

        # ensure pool
        while len(self._tile_widgets) < len(self._tiles_data):
            t = Tile()
            t.clicked.connect(self.tile_clicked)
            t.set_geometry(self._tile_w, self._tile_h)
            self._tile_widgets.append(t)

        # clear grid (keep widgets alive)
        while self.grid.count():
            it = self.grid.takeAt(0)
            w = it.widget()
            if w:
                w.setParent(None)

        # place
        for i, td in enumerate(self._tiles_data):
            r = i // self._n_cols
            c = i % self._n_cols

            w = self._tile_widgets[i]
            w.setParent(self.inner)
            w.set_geometry(self._tile_w, self._tile_h)
            w.set_content(td["h5_path"], td["row_idx"], td["rgb"], td.get("label", ""), td.get("tooltip", ""))

            self.grid.addWidget(w, r, c)
            self._key_to_widget[(td["h5_path"], int(td["row_idx"]))] = w

        # hide unused pooled widgets
        for j in range(len(self._tiles_data), len(self._tile_widgets)):
            self._tile_widgets[j].setParent(None)

        self.inner.adjustSize()

    def set_tile_label(self, h5_path: str, row_idx: int, new_label: str):
        w = self._key_to_widget.get((h5_path, int(row_idx)))
        if w:
            w.set_label(new_label)

    def tiles(self) -> List[Dict]:
        return self._tiles_data