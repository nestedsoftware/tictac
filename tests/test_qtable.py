import pytest

import numpy as np
import random
from collections import deque

from tictac.board import Board, CELL_X, CELL_O, new_board, play_random_move
from tictac.qtable import (INITIAL_Q_VALUES_FOR_O, INITIAL_Q_VALUES_FOR_X,
                           QTable, choose_move_index, create_training_player,
                           play_training_game, get_move_average_q_value_pairs)


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)


def test_get_q_values_initial_x_turn():
    b = np.array([[1, 0,  0],
                  [1, 0, -1],
                  [1, 0,  0]]).flatten()

    q_table = QTable()

    q_values = q_table.get_q_values(Board(b))

    expected_q_values = {1: INITIAL_Q_VALUES_FOR_X, 2: INITIAL_Q_VALUES_FOR_X,
                         4: INITIAL_Q_VALUES_FOR_X, 7: INITIAL_Q_VALUES_FOR_X,
                         8: INITIAL_Q_VALUES_FOR_X}

    assert q_values == expected_q_values


def test_get_q_values_initial_o_turn():
    b = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0,  0]]).flatten()

    q_table = QTable()

    q_values = q_table.get_q_values(Board(b))

    expected_q_values = {1: INITIAL_Q_VALUES_FOR_O, 4: INITIAL_Q_VALUES_FOR_O,
                         7: INITIAL_Q_VALUES_FOR_O, 8: INITIAL_Q_VALUES_FOR_O}

    assert q_values == expected_q_values


def test_choose_move_index_1st_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).flatten()

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, 1)
    q_table.update_q_value(board, 2, 0.5)

    move_index = choose_move_index([q_table], board, 0)

    assert move_index == 1


def test_choose_move_index_2nd_move():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).flatten()

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, 0.5)
    q_table.update_q_value(board, 2, 1)

    action_index = choose_move_index([q_table], board, 0)

    assert action_index == 2


def test_choose_move_index_with_transformation():
    b_2d = np.array([[1,  0,  0],
                     [1, -1,  1],
                     [-1, 1, -1]])

    b = b_2d.flatten()

    board = Board(b)

    q_table = QTable()
    q_table.update_q_value(board, 1, -1)
    q_table.update_q_value(board, 2, 1)

    b_transformed = np.rot90(b_2d, 2).flatten()

    board_transformed = Board(b_transformed)

    move_index = choose_move_index([q_table], board_transformed, 0)

    assert move_index == 6


def test_play_training_game_x_player():
    q_table = QTable()
    move_history = deque()
    q_table_player = CELL_X
    x_strategy = create_training_player([q_table], move_history, 0)
    o_strategy = play_random_move

    play_training_game([q_table], move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_X
    first_board = np.copy(new_board)

    val = 0.9 * 0.81
    expected_move_indexes_and_q_values = {0: val,  1: init, 2: val,
                                          3: init, 4: init, 5: init,
                                          6: val,  7: init, 8: val}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[0] = CELL_X
    second_board[7] = CELL_O

    val = 0.9 * 0.9
    expected_move_indexes_and_q_values = {1: val, 2: init,
                                          3: init, 4: init, 5: init,
                                          6: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    third_board = np.copy(second_board)
    third_board[1] = CELL_X
    third_board[5] = CELL_O

    val = 0.9 * 1.0
    expected_move_indexes_and_q_values = {2: val,
                                          3: init, 4: init,
                                          6: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(third_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    move_history = deque()
    x_strategy = create_training_player([q_table], move_history, 0)
    play_training_game([q_table], move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_X
    first_board = np.copy(new_board)

    val = 0.1 * (0.81 * 0.9) + 0.9 * (0.9 * (0.9 * 0.81))
    expected_move_indexes_and_q_values = {0: val,  1: init, 2: val,
                                          3: init, 4: init, 5: init,
                                          6: val,  7: init, 8: val}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[0] = CELL_X
    second_board[1] = CELL_O

    val = 0.9 * (0.9 * 0.81)
    expected_move_indexes_and_q_values = {2: val,
                                          3: init, 4: init, 5: init,
                                          6: init, 7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    third_board = np.copy(second_board)
    third_board[2] = CELL_X
    third_board[5] = CELL_O

    val = 0.9 * 0.81
    expected_move_indexes_and_q_values = {3: val, 4: init,
                                          6: init, 7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(third_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    fourth_board = np.copy(third_board)
    fourth_board[3] = CELL_X
    fourth_board[8] = CELL_O

    val = 0.81
    expected_move_indexes_and_q_values = {4: val,
                                          6: init, 7: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(fourth_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    fifth_board = np.copy(fourth_board)
    fifth_board[4] = CELL_X
    fifth_board[7] = CELL_O

    val = 0.9
    expected_move_indexes_and_q_values = {6: val}

    move_indexes_and_q_values = q_table.get_q_values(Board(fifth_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values


def test_play_training_game_o_player():
    q_table = QTable()
    move_history = deque()
    q_table_player = CELL_O
    x_strategy = play_random_move
    o_strategy = create_training_player([q_table], move_history, 0)

    play_training_game([q_table], move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_O
    first_board = np.copy(new_board)
    first_board[6] = CELL_X

    val = 0.9 * 0.81
    expected_move_indexes_and_q_values = {0: val,  1: init, 2: init,
                                          3: init, 4: init, 5: init,
                                          7: init, 8: val}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[0] = CELL_O
    second_board[8] = CELL_X

    val = 0.9 * 0.9
    expected_move_indexes_and_q_values = {1: val,  2: init,
                                          3: init, 4: init, 5: init,
                                          7: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    third_board = np.copy(second_board)
    third_board[1] = CELL_O
    third_board[5] = CELL_X

    val = 0.9 * 1.0
    expected_move_indexes_and_q_values = {2: val,
                                          3: init, 4: init,
                                          7: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(third_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    move_history = deque()
    o_strategy = create_training_player([q_table], move_history, 0)
    play_training_game([q_table], move_history, q_table_player, x_strategy,
                       o_strategy, 0.9, 1)

    init = INITIAL_Q_VALUES_FOR_O
    first_board = np.copy(new_board)
    first_board[0] = CELL_X

    val = (1 - 0.9) * (0.9 * 0.81) + (0.9 * 0.0)
    expected_move_indexes_and_q_values = {1: init, 2: val,
                                          3: init, 4: init, 5: init,
                                          6: val,  7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(first_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values

    second_board = np.copy(first_board)
    second_board[2] = CELL_O
    second_board[4] = CELL_X

    val = 0.9 * 0
    expected_move_indexes_and_q_values = {1: val,
                                          3: init, 5: init,
                                          6: init, 7: init, 8: init}

    move_indexes_and_q_values = q_table.get_q_values(Board(second_board))

    assert move_indexes_and_q_values == expected_move_indexes_and_q_values


def test_get_move_average_q_value_pairs():
    qtable_a = QTable()
    qtable_b = QTable()

    b_2d = np.array([[1,  0,  0],
                     [1, -1,  1],
                     [-1, 1, -1]])

    b = b_2d.flatten()

    board = Board(b)

    qtable_a.update_q_value(board, 1, 0.0)
    qtable_a.update_q_value(board, 2, 1.0)

    qtable_b.update_q_value(board, 1, -0.5)
    qtable_b.update_q_value(board, 2, 0.5)

    pairs = get_move_average_q_value_pairs([qtable_a, qtable_b], board)

    assert pairs == [(1, -0.25), (2, 0.75)]


def test_update_q_value():
    qtable = QTable()

    b_2d = np.array([[1.0,  0.0,  0.0],
                     [1.0, -1.0,  0.0],
                     [0.0,  1.0, -1.0]])
    b = b_2d.flatten()

    board = Board(b)

    qvalues = qtable.get_q_values(board)

    init = INITIAL_Q_VALUES_FOR_O

    expected_qvalues = {1: init, 2: init, 5: init, 6: init}

    assert qvalues == expected_qvalues

    b_rot90_flipud_2d = np.flipud(np.rot90(b_2d))
    b_rot90_flipud = b_rot90_flipud_2d.flatten()

    board_rot90_flipud = Board(b_rot90_flipud)

    qtable.update_q_value(board_rot90_flipud, 2, 0.8)
    qtable.update_q_value(board_rot90_flipud, 7, 0.7)

    assert len(qtable.qtable.cache) == 2

    expected_qvalues = {1: init, 2: init, 5: 0.7, 6: 0.8}

    qvalues = qtable.get_q_values(board)

    assert qvalues == expected_qvalues

    expected_qvalues = {2: 0.8, 3: init, 6: init, 7: 0.7}

    qvalues = qtable.get_q_values(board_rot90_flipud)

    assert qvalues == expected_qvalues
