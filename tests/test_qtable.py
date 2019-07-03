import pytest

import numpy as np

import random

from tictac.board import Board
from tictac.qtable import QTable, choose_move_index


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)


def test_get_q_values_initial():
    b = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0,  0]]).reshape(1, 9)[0]

    q_table = QTable()

    q_values = q_table.get_q_values(Board(b))

    expected_q_values = {1: 0.01, 4: 0.01, 7: 0.01, 8: 0.01}

    assert q_values == expected_q_values


def test_get_action_index_choose_1st_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).reshape(1, 9)[0]

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 0, 1)
    q_table.update_q_value(board, 1, 0.5)

    move_index = choose_move_index(q_table, board, 0)

    assert move_index == 0


def test_get_action_index_choose_2nd_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).reshape(1, 9)[0]

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, 0.5)
    q_table.update_q_value(board, 2, 1)

    action_index = choose_move_index(q_table, board, 0)

    assert action_index == 2
