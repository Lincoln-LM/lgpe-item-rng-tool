"""Widget for the item prediction tab in the main window"""

import numpy as np
from qtpy.QtWidgets import (
    QVBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QHBoxLayout,
    QLineEdit,
    QSpinBox,
    QTableWidget,
    QHeaderView,
    QTableWidgetItem,
)
from qtpy.QtGui import QRegularExpressionValidator
from qtpy import QtCore

from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection

from .range_widget import RangeWidget

# TODO: support other item tables
AREAS = {
    "Cerulean Cave Balls": (
        (50, "Poke Ball"),
        (100, "Great Ball"),
        (125, "Ultra Ball"),
        (1, "Master Ball"),
        (25, "10 Ultra Balls"),
    )
}


class ItemPredictionTab(QWidget):
    """QWidget for the item prediction tab in the main window"""

    def __init__(self) -> None:
        super().__init__()
        self.setup_widgets()

    def on_search_clicked(self) -> None:
        """Called when the search button is clicked"""
        rng = Xoroshiro128PlusRejection(
            np.uint64(int(self.state_0_input.text() or "0", 16)),
            np.uint64(int(self.state_1_input.text() or "0", 16)),
        )
        if rng.state[0] == 0 and rng.state[1] == 0:
            rng.state[1] = 1
            self.state_1_input.setText(f"{rng.state[1]:016X}")
        test_rng_menu = Xoroshiro128PlusRejection(0)
        test_rng_item = Xoroshiro128PlusRejection(0)
        item_table = AREAS[self.area_selector.currentText()]
        item_sum = sum(item[0] for item in item_table)
        advance_range = self.advance_range.get_range()
        rng.advance(advance_range.start)
        self.results.setRowCount(0)
        for advance in advance_range:
            test_rng_menu.re_init(rng.state[0], rng.state[1])
            rand_2 = rng.next_rand(2)
            for menu_open in range(self.max_menu_opens.value()):
                test_rng_item.re_init(test_rng_menu.state[0], test_rng_menu.state[1])
                item_rand = test_rng_item.next_rand(item_sum)
                for item in item_table:
                    if item_rand < item[0]:
                        break
                    item_rand -= item[0]
                # TODO: specific filtering
                if item[1] == "Master Ball":
                    row_i = self.results.rowCount()
                    self.results.insertRow(row_i)
                    row = (
                        str(advance),
                        str(menu_open),
                        str(rand_2),
                    )
                    for j, value in enumerate(row):
                        item = QTableWidgetItem()
                        item.setData(QtCore.Qt.EditRole, value)
                        self.results.setItem(row_i, j, item)
                    break
                test_rng_menu.next_rand(121)
                test_rng_menu.next_rand(2)

    def setup_widgets(self) -> None:
        """Construct widgets"""
        self.main_layout = QVBoxLayout(self)
        self.area_selector = QComboBox()
        self.area_selector.addItems(AREAS.keys())

        self.state_holder = QWidget()
        self.state_layout = QHBoxLayout(self.state_holder)
        self.state_label = QLabel("RNG State:")
        self.state_layout.addWidget(self.state_label)
        self.state_0_input = QLineEdit()
        self.state_0_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[0-9a-fA-F]{0,16}"))
        )
        self.state_layout.addWidget(self.state_0_input)
        self.state_1_input = QLineEdit()
        self.state_1_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[0-9a-fA-F]{0,16}"))
        )
        self.state_layout.addWidget(self.state_1_input)

        self.advance_range = RangeWidget(0, 1 << 30, "Advance Range:")
        self.advance_range.max_entry.setValue(1000)
        self.max_menu_opens_holder = QWidget()
        self.max_menu_opens_layout = QHBoxLayout(self.max_menu_opens_holder)
        self.max_menu_opens_label = QLabel("Max Partner Menu Opens:")
        self.max_menu_opens_layout.addWidget(self.max_menu_opens_label)
        self.max_menu_opens = QSpinBox()
        self.max_menu_opens.setValue(50)
        self.max_menu_opens.setMinimum(0)
        self.max_menu_opens.setMaximum(1000000)
        self.max_menu_opens_layout.addWidget(self.max_menu_opens)

        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.on_search_clicked)
        self.results = QTableWidget()
        self.results.setColumnCount(3)
        self.results.setHorizontalHeaderLabels(
            ["Advance", "Menus until Master Ball", "Rand(2)"]
        )
        self.results.verticalHeader().setVisible(False)
        header = self.results.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)

        self.main_layout.addWidget(self.area_selector)
        self.main_layout.addWidget(self.state_holder)
        self.main_layout.addWidget(self.advance_range)
        self.main_layout.addWidget(self.max_menu_opens_holder)
        self.main_layout.addWidget(self.search_button)
        self.main_layout.addWidget(self.results)
