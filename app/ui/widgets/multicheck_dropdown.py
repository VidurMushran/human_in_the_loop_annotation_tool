from PyQt5.QtWidgets import QToolButton, QMenu, QAction
from PyQt5.QtCore import pyqtSignal


class MultiCheckDropdown(QToolButton):
    changed = pyqtSignal()

    def __init__(self, title="Augmentations", parent=None):
        super().__init__(parent)
        self._base_title = title
        self.setText(title)
        self.setPopupMode(QToolButton.InstantPopup)

        self._menu = QMenu(self)
        self.setMenu(self._menu)
        self._actions = []

    def add_option(self, name: str, checked: bool = False):
        act = QAction(name, self._menu)
        act.setCheckable(True)
        act.setChecked(bool(checked))
        act.toggled.connect(lambda _: self._update_text())
        act.toggled.connect(lambda _: self.changed.emit())
        self._menu.addAction(act)
        self._actions.append(act)
        self._update_text()

    def selected(self):
        return [a.text() for a in self._actions if a.isChecked()]

    def _update_text(self):
        sel = self.selected()
        if not sel:
            self.setText(f"{self._base_title}: none")
        else:
            self.setText(f"{self._base_title}: " + ", ".join(sel))
