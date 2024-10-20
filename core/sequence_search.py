"""
Utilty code to search partner observations

Initial implementation: https://gist.github.com/Lincoln-LM/a7be1e81171218775399dda3da963030
"""

from itertools import combinations_with_replacement
import numpy as np
import numba
from numba_pokemon_prngs.xorshift import Xoroshiro128PlusRejection
from numba_progress.numba_atomic import atomic_add
from qtpy import QtCore

NUM_THREADS = numba.config.NUMBA_NUM_THREADS


@numba.njit
def test_sequence(
    rng: Xoroshiro128PlusRejection, observations, jumps, result_count, results
):
    """Test a given combination to see if it produces the expected sequence"""
    # construct a matrix that maps from an initial state to the observations
    mat = np.zeros((64, 64), np.uint8)
    for bit in range(64):
        seed_0 = 1 << bit if bit < 64 else 0
        rng.re_init(seed_0, 0)
        for i in range(64):
            rng.next()
            for j in jumps:
                if i == j:
                    rng.next()
            mat[bit, i] = rng.next_rand(2)
    # construct vector that stores the influence of 0x82A2B175229D6A5B on the observations
    const_influence = np.empty(64, np.uint8)
    rng.re_init(0, np.uint64(0x82A2B175229D6A5B))
    for i in range(64):
        rng.next()
        for j in jumps:
            if i == j:
                rng.next()
        const_influence[i] = rng.next_rand(2)

    # invert the matrix to map from observations to the state
    inverse = np.empty(64, np.uint64)
    # identity
    for bit in range(64):
        inverse[bit] = 1 << bit

    rank = 0
    pivots = []
    for col in range(64):
        for row in range(rank, 64):
            if mat[row, col]:
                for other_row in range(64):
                    if (other_row != row) and mat[other_row, col]:
                        mat[other_row] ^= mat[row]
                        inverse[other_row] ^= inverse[row]
                temp = np.copy(mat[row])
                mat[row] = mat[rank]
                mat[rank] = temp
                temp = inverse[row]
                inverse[row] = inverse[rank]
                inverse[rank] = temp
                pivots.append(col)
                rank += 1
                break

    # store the nullbasis in the event that the observations are not determinantive
    nullbasis = np.copy(inverse[rank:])
    # undo pivots
    for i in range(rank - 1, -1, -1):
        pivot = pivots[i]
        temp = inverse[i]
        inverse[i] = inverse[pivot]
        inverse[pivot] = temp

    # (observations - const_influence) @ inverse
    principal_result = np.uint64(0)
    for i in range(64):
        if observations[i] ^ const_influence[i]:
            principal_result ^= inverse[i]

    # loop over other solutions
    for i in range(1 << len(nullbasis)):
        result = principal_result
        for bit in range(64):
            if i == 0:
                break
            if i & 1:
                result ^= nullbasis[bit]
            i >>= 1

        # test result
        rng.re_init(result, np.uint64(0x82A2B175229D6A5B))
        valid = True
        for observation in observations:
            rng.next_rand(121)
            valid &= (rng.next() & 1) == observation
            if not valid:
                break
        if valid:
            results[atomic_add(result_count, 0, 1)] = result


@numba.njit(
    numba.void(
        numba.uint8[:],
        numba.int8[:, :],
        numba.uint64[:],
        numba.uint64[:],
        numba.uint64[:],
    ),
    nogil=True,
    parallel=True,
)
def search_sequences(
    observations: np.ndarray,
    combinations: np.ndarray,
    result_count: np.ndarray,
    progress: np.ndarray,
    results: np.ndarray,
) -> None:
    """Test all given combinations of jumps"""
    for thread_i in numba.prange(NUM_THREADS):
        rng = Xoroshiro128PlusRejection(0, 0)
        ofs = combinations.shape[0] // NUM_THREADS * thread_i
        end = combinations.shape[0] // NUM_THREADS * (thread_i + 1)
        if thread_i == NUM_THREADS - 1:
            end = combinations.shape[0]
        for i in range(ofs, end):
            test_sequence(rng, observations, combinations[i], result_count, results)
            atomic_add(progress, 0, 1)


class SequenceSearchThread(QtCore.QThread):
    """Thread to do the actual work of searching a partner sequence"""

    def __init__(self, observations: list[int], max_jumps: int) -> None:
        super().__init__()
        self.observations = np.array(observations, np.uint8)

        combinations = []
        for jump_count in range(max_jumps + 1):
            for jumps in combinations_with_replacement(range(64), jump_count):
                combinations.append(
                    jumps + tuple(-1 for _ in range(max_jumps - jump_count))
                )

        self.combinations = np.array(combinations, dtype=np.int8)

        self.result_count = np.zeros(1, np.uint64)
        self.progress = np.zeros(1, np.uint64)
        # huge overestimation of the number of results
        self.results = np.zeros(0x10000, np.uint64)

    def run(self) -> None:
        """Thread work"""
        search_sequences(
            self.observations,
            self.combinations,
            self.result_count,
            self.progress,
            self.results,
        )
