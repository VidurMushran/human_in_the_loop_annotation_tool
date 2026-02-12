from __future__ import annotations
from typing import List
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit

class LabelEditor(QWidget):
    active_label_changed = pyqtSignal(str)
    labels_changed = pyqtSignal(list)

    def __init__(self, labels: List[str], active: str):
        super().__init__()
        self.labels = list(labels)

        lay = QHBoxLayout(self)

        lay.addWidget(QLabel("Active label:"))
        self.combo = QComboBox()
        self.combo.addItems(self.labels)
        if active in self.labels:
            self.combo.setCurrentText(active)
        self.combo.currentTextChanged.connect(self.active_label_changed)
        lay.addWidget(self.combo)

        self.new_label = QLineEdit()
        self.new_label.setPlaceholderText("Add new label name…")
        lay.addWidget(self.new_label)

        btn = QPushButton("Add")
        btn.clicked.connect(self.add_label)
        lay.addWidget(btn)

        lay.addStretch(1)

    def add_label(self):
        s = self.new_label.text().strip()
        if not s or s in self.labels:
            return
        self.labels.append(s)
        self.combo.addItem(s)
        self.new_label.clear()
        self.labels_changed.emit(self.labels)
