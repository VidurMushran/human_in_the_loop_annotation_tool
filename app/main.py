import sys
from PyQt5.QtWidgets import QApplication
from app.ui.main_window import MainWindow
from app.utils.logging import setup_logging

def main():
    setup_logging()
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
