"""Main script for lgpe-item-rng-tool"""

import sys
from core.main_window import MainWindow
from qtpy.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = MainWindow()
    window.show()
    window.setFocus()

    sys.exit(app.exec())
