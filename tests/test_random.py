import numpy as np

from tictac import random


def test_get_random_valid_move():
    board = np.array([0, -1, 0, 0, -1, 0, 1, 0, 1])
    move = random.get_random_valid_move(board)

    assert move in [0, 2, 3, 5, 7]
    assert move not in [1, 4, 6, 8]
