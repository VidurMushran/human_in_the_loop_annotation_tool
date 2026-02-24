from PyQt5.QtWidgets import QMainWindow, QTabWidget
from app.utils.config import load_config, save_config
from app.ui.annotate_tab import AnnotateTab
from app.ui.train_tab import TrainTab
from app.ui.metrics_tab import MetricsTab
from junk_gui_app.app.ui.comparison_tab import ComparisonTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Junk ↔ Cells Human-in-the-loop Trainer (PyQt)")
        self.resize(1400, 900)

        self.cfg = load_config()

        tabs = QTabWidget()
        self.annotate_tab = AnnotateTab(self.cfg)
        self.train_tab = TrainTab(self.cfg, self.annotate_tab)
        self.metrics_tab = MetricsTab(self.cfg, self.annotate_tab)
        self.comparison_tab = ComparisonTab(self.cfg, self.annotate_tab)

        tabs.addTab(self.annotate_tab, "Annotate")
        tabs.addTab(self.train_tab, "Train / Score")
        tabs.addTab(self.metrics_tab, "Metrics")
        tabs.addTab(self.comparison_tab, "Compare Models")
        
        self.setCentralWidget(tabs)

    def closeEvent(self, e):
        save_config(self.cfg)
        super().closeEvent(e)
