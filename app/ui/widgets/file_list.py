from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QListWidget, QListWidgetItem
from PyQt5.QtCore import Qt

class FileListWidget(QWidget):
    selection_changed = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.list = QListWidget()
        self.list.itemChanged.connect(self._emit)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("HDF5 files (check to include in gallery):"))
        layout.addWidget(self.list)

    def set_files(self, files):
        self.list.blockSignals(True)
        self.list.clear()
        for group, path in files:
            fname = path.split("/")[-1]
            it = QListWidgetItem(f"[{group}] {fname}")
            it.setData(Qt.UserRole, path)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.list.addItem(it)
        self.list.blockSignals(False)

    def _emit(self):
        sel = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.checkState() == Qt.Checked:
                sel.append(it.data(Qt.UserRole))
        self.selection_changed.emit(sel)
