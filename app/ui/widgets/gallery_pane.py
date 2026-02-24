from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from .gallery import GalleryWidget
from typing import List, Dict

class GalleryPane(QWidget):
    """A header + a GalleryWidget."""
    def __init__(self, title: str, tile_clicked_cb=None):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)

        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.title)

        self.gallery = GalleryWidget()
        if tile_clicked_cb:
            self.gallery.tile_clicked.connect(tile_clicked_cb)
        lay.addWidget(self.gallery, 1)

    def set_title(self, title: str):
        self.title.setText(title)

    def set_layout(self, n_cols: int, tile_h: int, tile_w: int):
        self.gallery.set_layout(n_cols=n_cols, tile_h=tile_h, tile_w=tile_w)

    def set_tiles(self, tiles: List[Dict]):
        self.gallery.set_tiles(tiles)

    def set_tile_label(self, h5_path: str, row_idx: int, new_label: str):
        self.gallery.set_tile_label(h5_path, row_idx, new_label)

    def tiles(self) -> List[Dict]:
        return self.gallery.tiles()