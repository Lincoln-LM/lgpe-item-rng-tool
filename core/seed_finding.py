"""QWidget for the seed finding tab in the main window"""

from enum import IntEnum
from math import ceil, log2
from time import sleep
from qtpy.QtWidgets import (
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QListWidget,
    QComboBox,
    QCheckBox,
    QMenu,
    QSpinBox,
)
from qtpy.QtGui import QRegularExpressionValidator
from qtpy import QtCore

import numpy as np
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection

from .eta_progress_bar import ETAProgressBar
from .range_widget import RangeWidget
from .sequence_search import SequenceSearchThread
from .matrix_utility import *


class ObservationType(IntEnum):
    """Enum for observation types"""

    PARTNER_OBSERVATIONS = 0
    PARTNER_OBSERVATIONS_REIDENTIFICATION = 1
    CANDY_OBSERVATIONS_INITAL = 2
    CANDY_OBSERVATIONS_ARBITRARY = 3


class FullSearchThread(QtCore.QThread):
    """Thread for full seed search"""

    progress = QtCore.Signal(int)
    result = QtCore.Signal(object)
    total_work = QtCore.Signal(int)

    def __init__(
        self, observations: list[int], observation_type: ObservationType, max_jumps: int
    ) -> None:
        super().__init__()
        self.observations = observations
        self.observation_type = observation_type
        self.max_jumps = max_jumps

    def run(self) -> None:
        """Thread work"""
        if self.observation_type == ObservationType.PARTNER_OBSERVATIONS:
            sequence_search_thread = SequenceSearchThread(
                self.observations, self.max_jumps
            )
            self.total_work.emit(sequence_search_thread.combinations.shape[0])
            sequence_search_thread.start()
            result_count = 0
            results = set()
            while sequence_search_thread.isRunning():
                sleep(0.1)
                if sequence_search_thread.result_count[0] > result_count:
                    for result in sequence_search_thread.results[
                        result_count : sequence_search_thread.result_count[0]
                    ]:
                        if result not in results:
                            results.add(result)
                            self.result.emit((result, 0x82A2B175229D6A5B))
                    result_count = sequence_search_thread.result_count[0]
                self.progress.emit(sequence_search_thread.progress[0])
        elif self.observation_type == ObservationType.CANDY_OBSERVATIONS_INITAL:
            self.total_work.emit(1)
            mat = np.zeros((64, len(self.observations)), np.uint8)
            rng = Xoroshiro128PlusRejection(0)
            for bit in range(64):
                rng.re_init(np.uint64(1 << bit), 0)
                mat[bit, :] = tuple(
                    rng.next_rand(2) for _ in range(len(self.observations))
                )
            inverse, nullbasis = generalized_inverse(mat)
            rng.re_init(0)
            principal_seed = bit_vector_to_int(
                (
                    tuple(
                        observation ^ rng.next_rand(2)
                        for observation in self.observations
                    )
                    @ inverse
                )
                & 1
            )
            for i in range(1 << nullbasis.shape[0]):
                seed = principal_seed
                nullbasis_iter = iter(nullbasis)
                while i:
                    nb_vector = next(nullbasis_iter)
                    if i & 1:
                        seed ^= bit_vector_to_int(nb_vector)
                    i >>= 1
                self.result.emit((seed, 0x82A2B175229D6A5B))
            self.progress.emit(1)
        elif self.observation_type == ObservationType.CANDY_OBSERVATIONS_ARBITRARY:
            self.total_work.emit(1)
            mat = np.zeros((128, len(self.observations)), np.uint8)
            rng = Xoroshiro128PlusRejection(0)
            for bit in range(64):
                rng.re_init(np.uint64(1 << bit), 0)
                mat[bit, :] = tuple(
                    rng.next_rand(2) for _ in range(len(self.observations))
                )
            for bit in range(64):
                rng.re_init(0, np.uint64(1 << bit))
                mat[bit + 64, :] = tuple(
                    rng.next_rand(2) for _ in range(len(self.observations))
                )
            inverse, nullbasis = generalized_inverse(mat)
            principal_seed = bit_vector_to_int((self.observations @ inverse) & 1)
            for i in range(1 << nullbasis.shape[0]):
                seed = principal_seed
                nullbasis_iter = iter(nullbasis)
                while i:
                    nb_vector = next(nullbasis_iter)
                    if i & 1:
                        seed ^= bit_vector_to_int(nb_vector)
                    i >>= 1
                self.result.emit((seed & 0xFFFFFFFFFFFFFFFF, seed >> 64))
            self.progress.emit(1)


def test_result(
    result: tuple[int, int],
    observations: list[int],
    observation_type: ObservationType,
) -> bool:
    """Test if a state produces the given observations"""
    rng = Xoroshiro128PlusRejection(np.uint64(result[0]), np.uint64(result[1]))
    if observation_type in (
        ObservationType.PARTNER_OBSERVATIONS,
        ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
    ):
        for observation in observations:
            rng.next_rand(121)
            if observation != rng.next_rand(2):
                return False
    else:
        return all(observation == rng.next_rand(2) for observation in observations)

    return True


class SeedFindingTab(QWidget):
    """QWidget for the seed finding tab in the main window"""

    def __init__(self, rng_tab) -> None:
        super().__init__()
        self.current_results = []
        self.full_search_thread = None
        self.rng_tab = rng_tab
        self.setup_widgets()

    def on_observations_changed(self) -> None:
        """Called whenever any entered observations are changed"""

        if self.observation_type.currentData() == ObservationType.PARTNER_OBSERVATIONS:
            observations = self.partner_observations_input.text()
            self.partner_observations_input_label.setText(
                f"Partner Observations (d/p) ({len(observations)}/64+):"
            )
            # only allow search if all 64 bits of info are known
            if len(observations) >= 64:
                self.search_button.setEnabled(True)
            else:
                self.search_button.setEnabled(False)
        elif (
            self.observation_type.currentData()
            == ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION
        ):
            observations = self.partner_observations_input.text()
            reident_range = self.reident_range.get_range()
            needed_observations = ceil(log2(reident_range.stop - reident_range.start))
            self.partner_observations_input_label.setText(
                f"Partner Observations (d/p) ({len(observations)}/{needed_observations}):"
            )
            self.search_button.setEnabled(True)
        else:
            is_initial_search = (
                self.observation_type.currentData()
                == ObservationType.CANDY_OBSERVATIONS_INITAL
            )
            observations = self.candy_observations_input.text()
            self.candy_observations_input_label.setText(
                f"Candy Observations (q/s) ({len(observations)}/{64 if is_initial_search else 128}+):"
            )
            # only allow search if < 16 bits of info is needed
            if len(observations) > 64 - 16 if is_initial_search else 128 - 16:
                self.search_button.setEnabled(True)
            else:
                self.search_button.setEnabled(False)

    def on_search_finished(self) -> None:
        """Called when the search is finished"""
        if self.current_results:
            self.search_button.setText("Search From Results")
            self.reset_search_button.setVisible(True)
        else:
            self.search_button.setText("Search")
            self.reset_search_button.setVisible(False)

    def on_search_button_clicked(self) -> None:
        """Called when the search button is clicked"""

        self.result_list.clear()
        observation_type = self.observation_type.currentData()
        observations_input = (
            self.partner_observations_input
            if observation_type
            in (
                ObservationType.PARTNER_OBSERVATIONS,
                ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
            )
            else self.candy_observations_input
        )
        observations = list(
            int(observation in "ps") for observation in observations_input.text()
        )
        if (
            not self.current_results
            and observation_type
            == ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION
        ):
            rng = Xoroshiro128PlusRejection(
                np.uint64(int(self.reident_state_0_input.text() or "0", 16)),
                np.uint64(int(self.reident_state_1_input.text() or "0", 16)),
            )
            reident_range = self.reident_range.get_range()
            rng.advance(reident_range.start)
            rng.previous()
            self.current_results = [
                tuple(rng.state) for _ in (rng.next() for _ in reident_range)
            ]
        if self.current_results:
            self.current_results = [
                result
                for result in self.current_results
                if test_result(result, observations, observation_type)
            ]
            for result in self.current_results:
                rng = Xoroshiro128PlusRejection(
                    np.uint64(result[0]), np.uint64(result[1])
                )
                for _ in observations:
                    if observation_type in (
                        ObservationType.PARTNER_OBSERVATIONS,
                        ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
                    ):
                        rng.next_rand(121)
                    rng.next_rand(2)
                result = tuple(rng.state)
                self.result_list.addItem(f"{result[0]:016X} {result[1]:016X}")
            self.on_search_finished()
        else:
            # TODO: cancel search
            def on_result(result: tuple[int, int]) -> None:
                # sanity check results
                if test_result(result, observations, observation_type):
                    self.current_results.append(result)
                    rng = Xoroshiro128PlusRejection(
                        np.uint64(result[0]), np.uint64(result[1])
                    )
                    for _ in observations:
                        if observation_type in (
                            ObservationType.PARTNER_OBSERVATIONS,
                            ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
                        ):
                            rng.next_rand(121)
                        rng.next_rand(2)
                    result = tuple(rng.state)
                    self.result_list.addItem(f"{result[0]:016X} {result[1]:016X}")

            self.full_search_thread = FullSearchThread(
                observations, observation_type, self.max_jumps_input.value()
            )
            self.full_search_thread.total_work.connect(self.progress_bar.setMaximum)
            self.full_search_thread.progress.connect(self.progress_bar.setValue)
            self.full_search_thread.result.connect(on_result)
            self.full_search_thread.finished.connect(self.on_search_finished)
            self.full_search_thread.start()

    def on_reset_search_button_clicked(self) -> None:
        """Called when the reset search button is clicked"""
        self.result_list.clear()
        self.reset_search_button.setVisible(False)
        self.search_button.setText("Search")
        self.current_results = []

    def result_context_menu_requested(self, event) -> None:
        """Called when the result list context menu is requested"""
        if not self.current_results:
            return
        menu = QMenu(self)

        def send_to_reidentification() -> None:
            seeds = self.result_list.currentItem().text().split(" ")
            self.reident_state_0_input.setText(seeds[0])
            self.reident_state_1_input.setText(seeds[1])

        def send_to_rng_state() -> None:
            seeds = self.result_list.currentItem().text().split(" ")
            self.rng_tab.state_0_input.setText(seeds[0])
            self.rng_tab.state_1_input.setText(seeds[1])

        menu.addAction("Send to reidentification", send_to_reidentification)
        menu.addAction("Send to rng state", send_to_rng_state)
        menu.popup(event.globalPos())

    def setup_widgets(self) -> None:
        """Construct seed finding tab widgets"""
        self.main_layout = QVBoxLayout(self)
        self.reidentification_check = QCheckBox("Reidentification")

        def on_reidentification_check_changed(reidentification: bool) -> None:
            self.observation_type.clear()
            if reidentification:
                self.observation_type.addItem(
                    "Partner Observations Reidentification",
                    ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
                )
            else:
                self.observation_type.addItem(
                    "Partner Observations Initial State",
                    ObservationType.PARTNER_OBSERVATIONS,
                )
                self.observation_type.addItem(
                    "Candy Observations Initial State",
                    ObservationType.CANDY_OBSERVATIONS_INITAL,
                )
                self.observation_type.addItem(
                    "Candy Observations Arbitrary State",
                    ObservationType.CANDY_OBSERVATIONS_ARBITRARY,
                )
            self.reident_range.setVisible(reidentification)
            self.reident_state_holder.setVisible(reidentification)

        self.reidentification_check.stateChanged.connect(
            on_reidentification_check_changed
        )
        self.observation_type = QComboBox()

        def on_observation_type_changed(_: int) -> None:
            self.candy_observations_input.setText("")
            self.partner_observations_input.setText("")
            is_partner_observations = self.observation_type.currentData() in (
                ObservationType.PARTNER_OBSERVATIONS,
                ObservationType.PARTNER_OBSERVATIONS_REIDENTIFICATION,
            )
            self.candy_observations_input_holder.setVisible(not is_partner_observations)
            self.partner_observations_input_holder.setVisible(is_partner_observations)
            self.max_jumps_input_holder.setVisible(
                self.observation_type.currentData()
                == ObservationType.PARTNER_OBSERVATIONS
            )

        self.observation_type.currentIndexChanged.connect(on_observation_type_changed)

        self.candy_observations_input_holder = QWidget()
        self.candy_observations_input_layout = QHBoxLayout(
            self.candy_observations_input_holder
        )
        self.candy_observations_input = QLineEdit()
        self.candy_observations_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[qsQS]*"))
        )
        self.candy_observations_input_label = QLabel("Candy Observations (q/s):")
        self.candy_observations_input_layout.addWidget(
            self.candy_observations_input_label
        )
        self.candy_observations_input_layout.addWidget(self.candy_observations_input)
        self.candy_observations_input.textChanged.connect(self.on_observations_changed)

        self.partner_observations_input_holder = QWidget()
        self.partner_observations_input_layout = QHBoxLayout(
            self.partner_observations_input_holder
        )
        self.partner_observations_input = QLineEdit()
        self.partner_observations_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[dpDP]*"))
        )
        self.partner_observations_input_label = QLabel("Partner Observations (d/p):")
        self.partner_observations_input_layout.addWidget(
            self.partner_observations_input_label
        )
        self.partner_observations_input_layout.addWidget(
            self.partner_observations_input
        )
        self.partner_observations_input.textChanged.connect(
            self.on_observations_changed
        )
        self.max_jumps_input_holder = QWidget()
        self.max_jumps_input_layout = QHBoxLayout(self.max_jumps_input_holder)
        self.max_jumps_input_label = QLabel("Max Jumps:")
        self.max_jumps_input_layout.addWidget(self.max_jumps_input_label)
        self.max_jumps_input = QSpinBox()
        self.max_jumps_input.setRange(0, 5)
        self.max_jumps_input.setValue(3)
        self.max_jumps_input_layout.addWidget(self.max_jumps_input)

        self.reident_state_holder = QWidget()
        self.reident_state_layout = QHBoxLayout(self.reident_state_holder)
        self.reident_state_label = QLabel("Reidentification State:")
        self.reident_state_layout.addWidget(self.reident_state_label)
        self.reident_state_0_input = QLineEdit()
        self.reident_state_0_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[0-9a-fA-F]{0,16}"))
        )
        self.reident_state_layout.addWidget(self.reident_state_0_input)
        self.reident_state_1_input = QLineEdit()
        self.reident_state_1_input.setValidator(
            QRegularExpressionValidator(QtCore.QRegularExpression("[0-9a-fA-F]{0,16}"))
        )
        self.reident_state_layout.addWidget(self.reident_state_1_input)
        self.reident_range = RangeWidget(0, 99999999, "Reidentification Range:")
        self.reident_range.max_entry.setValue(10000)
        self.reident_range.min_entry.valueChanged.connect(self.on_observations_changed)
        self.reident_range.max_entry.valueChanged.connect(self.on_observations_changed)

        on_observation_type_changed(None)
        on_reidentification_check_changed(self.reidentification_check.isChecked())

        self.search_buttons_holder = QWidget()
        self.search_buttons_layout = QHBoxLayout(self.search_buttons_holder)
        self.search_button = QPushButton("Search")
        self.search_button.setEnabled(False)
        self.search_button.clicked.connect(self.on_search_button_clicked)
        self.reset_search_button = QPushButton("Clear Results")
        self.reset_search_button.setVisible(False)
        self.reset_search_button.clicked.connect(self.on_reset_search_button_clicked)
        self.search_buttons_layout.addWidget(self.search_button)
        self.search_buttons_layout.addWidget(self.reset_search_button)

        self.progress_bar = ETAProgressBar()

        self.result_list = QListWidget()
        self.result_list.contextMenuEvent = self.result_context_menu_requested

        self.main_layout.addWidget(self.reidentification_check)
        self.main_layout.addWidget(self.observation_type)
        self.main_layout.addWidget(self.candy_observations_input_holder)
        self.main_layout.addWidget(self.partner_observations_input_holder)
        self.main_layout.addWidget(self.max_jumps_input_holder)
        self.main_layout.addWidget(self.reident_state_holder)
        self.main_layout.addWidget(self.reident_range)
        self.main_layout.addWidget(self.search_buttons_holder)
        self.main_layout.addWidget(self.progress_bar)
        self.main_layout.addWidget(self.result_list)
