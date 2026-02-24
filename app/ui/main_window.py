from PyQt5.QtWidgets import QMainWindow, QTabWidget
from .annotate_tab import AnnotateTab
from .train_tab import TrainTab
from .metrics_tab import MetricsTab
from .comparison_tab import ComparisonTab
from .cluster_tab import ClusterTab # NEW
from ..utils.config import AppConfig

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Junk Annotation Tool")
        self.resize(1400, 900)
        self.cfg = AppConfig()
        
        tabs = QTabWidget()
        self.annotate_tab = AnnotateTab(self.cfg)
        self.train_tab = TrainTab(self.cfg, self.annotate_tab)
        self.metrics_tab = MetricsTab(self.cfg, self.annotate_tab)
        self.comp_tab = ComparisonTab(self.cfg, self.annotate_tab)
        self.cluster_tab = ClusterTab(self.cfg, self.annotate_tab)
        
        tabs.addTab(self.annotate_tab, "Annotate")
        tabs.addTab(self.cluster_tab, "Cluster Assist")
        tabs.addTab(self.train_tab, "Train")
        tabs.addTab(self.metrics_tab, "Metrics")
        tabs.addTab(self.comp_tab, "Compare")
        
        self.setCentralWidget(tabs)