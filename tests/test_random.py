import numpy as np

from tictac.board import Board


def test_get_random_valid_move():
    b = np.array([0, -1, 0, 0, -1, 0, 1, 0, 1])

    move = Board(b).get_random_valid_move_index()

    assert move in [0, 2, 3, 5, 7]
