from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(QWidget):
    def __init__(self):
        super().__init__()
        fig = Figure(figsize=(4,4))
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvas(fig)
        lay = QVBoxLayout(self)
        lay.addWidget(self.canvas)

    def draw(self):
        self.canvas.draw()
