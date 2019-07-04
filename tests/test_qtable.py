import pytest

import numpy as np
import random
from collections import deque

from tictac.board import Board, CELL_X, CELL_O, new_board, play_random_move
from tictac.qtable import (INITIAL_Q_VALUES_FOR_O, INITIAL_Q_VALUES_FOR_X,
                           QTable, choose_move_index, create_play_for_training,
                           play_training_game)


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)


def test_get_q_values_initial_x_turn():
    b = np.array([[1, 0,  0],
                  [1, 0, -1],
                  [1, 0,  0]]).reshape(1, 9)[0]

    q_table = QTable()

    q_values = q_table.get_q_values(Board(b))

    expected_q_values = {1: INITIAL_Q_VALUES_FOR_X, 2: INITIAL_Q_VALUES_FOR_X,
                         4: INITIAL_Q_VALUES_FOR_X, 7: INITIAL_Q_VALUES_FOR_X,
                         8: INITIAL_Q_VALUES_FOR_X}

    assert q_values == expected_q_values


def test_get_q_values_initial_o_turn():
    b = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0,  0]]).reshape(1, 9)[0]

    q_table = QTable()

    q_values = q_table.get_q_values(Board(b))

    expected_q_values = {1: INITIAL_Q_VALUES_FOR_O, 4: INITIAL_Q_VALUES_FOR_O,
                         7: INITIAL_Q_VALUES_FOR_O, 8: INITIAL_Q_VALUES_FOR_O}

    assert q_values == expected_q_values


def test_choose_move_index_1st_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).reshape(1, 9)[0]

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 0, 1)
    q_table.update_q_value(board, 1, 0.5)

    move_index = choose_move_index(q_table, board, 0)

    assert move_index == 0


def test_choose_move_index_2nd_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).reshape(1, 9)[0]

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, 0.5)
    q_table.update_q_value(board, 2, 1)

    action_index = choose_move_index(q_table, board, 0)

    assert action_index == 2


def test_choose_move_index_with_transformation():
    b_2d = np.array([[1,  0,  0],
                     [1, -1,  1],
                     [-1, 1, -1]])

    b = b_2d.reshape(1, 9)[0]

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, -1)
    q_table.update_q_value(board, 2, 1)

    b_transformed = np.rot90(b_2d, 2).reshape(1, 9)[0]

    board_transformed = Board(b_transformed)

    move_index = choose_move_index(q_table, board_transformed, 0)

    assert move_index == 6


def test_play_training_game_x_player():
    q_table = QTable()
    move_history = deque()
    q_table_player = CELL_X
    x_strategy = create_play_for_training(q_table, move_history, 0)
    o_strategy = play_random_move

    play_training_game(q_table, move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_X
    first_board = np.copy(new_board)

    expected_move_indexes_and_q_values = {0: 0.81, 1: init, 2: init,
                                          3: init, 4: init, 5: init,
                                          6: init, 7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[0] = CELL_X
    second_board[7] = CELL_O

    expected_move_indexes_and_q_values = {1: 0.9, 2: init,
                                          3: init, 4: init, 5: init,
                                          6: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    third_board = np.copy(second_board)
    third_board[1] = CELL_X
    third_board[5] = CELL_O

    expected_move_indexes_and_q_values = {2: 1.0,
                                          3: init, 4: init,
                                          6: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(third_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values


def test_play_training_game_o_player():
    q_table = QTable()
    move_history = deque()
    q_table_player = CELL_O
    x_strategy = play_random_move
    o_strategy = create_play_for_training(q_table, move_history, 0)

    play_training_game(q_table, move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_O
    first_board = np.copy(new_board)
    first_board[6] = CELL_X

    qvalue = 0.7150000000000001
    expected_move_indexes_and_q_values = {0: qvalue, 1: init, 2: init,
                                          3: init, 4: init, 5: init,
                                          7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[0] = CELL_O
    second_board[8] = CELL_X

    qvalue = 0.8500000000000001
    expected_move_indexes_and_q_values = {1: qvalue, 2: init,
                                          3: init, 4: init, 5: init,
                                          7: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    third_board = np.copy(second_board)
    third_board[1] = CELL_O
    third_board[5] = CELL_X
    
    expected_move_indexes_and_q_values = {2: 1.0,
                                          3: init, 4: init,
                                          7: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(third_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values
