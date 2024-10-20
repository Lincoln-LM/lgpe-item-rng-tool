"""QWidget window for the main program"""

from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QTabWidget,
)

from .seed_finding import SeedFindingTab
from .item_prediction import ItemPredictionTab


class MainWindow(QWidget):
    """QWidget window for the main program"""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("LGPE Item RNG Tool")
        self.setup_widgets()

        self.show()

    def setup_widgets(self) -> None:
        """Construct main window widgets"""
        self.main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()

        rng_tab = ItemPredictionTab()
        self.tab_widget.addTab(SeedFindingTab(rng_tab), "Seed Finding")
        self.tab_widget.addTab(rng_tab, "Item Prediction")

        self.main_layout.addWidget(self.tab_widget)
