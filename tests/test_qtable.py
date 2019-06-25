import pytest

import numpy as np

import random

from tictac.board_cache import BoardCache
from tictac.qtable import get_q_values, choose_move


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)


def test_get_q_values_initial():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()

    q_values = get_q_values(q_table, board)

    assert np.array_equal(q_values, np.full(4, 0.01))


def test_get_action_index_choose_1st_move():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([1, 0.5]))

    action_index = choose_move(q_table, board, 0)

    assert action_index == 1


def test_get_action_index_choose_2nd_move():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([0.5, 1]))

    action_index = choose_move(q_table, board, 0)

    assert action_index == 2
