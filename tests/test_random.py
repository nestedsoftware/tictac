from numpy import array

from src.board import Board


def test_get_random_valid_move():
    b = array([0, -1, 0, 0, -1, 0, 1, 0, 1])
    assert Board(b).get_random_valid_move_index() in [0, 2, 3, 5, 7]
