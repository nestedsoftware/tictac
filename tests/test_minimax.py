import pytest

import numpy as np

from tictac.minimax import cache
from tictac.minimax import (get_position_value, get_move_value_pairs,
                            play_minimax_move)
from tictac.board import Board
from tictac.board import RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW
from tictac.board import get_symmetrical_board_orientations


@pytest.fixture(autouse=True)
def reset_cache():
    cache.reset()


def test_get_position_value_x_wins():
    b = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [1, 0,  0]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_X_WINS


def test_get_position_value_o_wins():
    b = np.array([[1, 0, -1],
                  [1, 0, -1],
                  [0, 1, -1]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_O_WINS


def test_get_position_value_draw():
    b = np.array([[1, -1,  1],
                  [1,  1, -1],
                  [-1, 1, -1]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_DRAW


def test_get_position_value_draw_is_best_case():
    b = np.array([[1, -1,  0],
                  [1,  1, -1],
                  [-1, 1, -1]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_DRAW


def test_get_position_value_o_wins_in_best_case_x_turn():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 0, -1]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_O_WINS


def test_get_position_value_o_wins_in_best_case_o_turn():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [0,  0, -1]]).flatten()

    value = get_position_value(Board(b))

    assert value == RESULT_O_WINS


def test_get_move_value_pairs_for_position_o_wins_in_best_case():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [0,  0, -1]]).flatten()

    move_value_pairs = get_move_value_pairs(Board(b))

    assert move_value_pairs == [(1, 1), (2, 1), (6, -1), (7, 1)]


def test_play_minimax_move_o_wins_in_best_case():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [0,  0, -1]]).flatten()

    result = play_minimax_move(Board(b)).board

    assert np.array_equal(result, np.array([[1,  0,  0],
                                            [1, -1,  1],
                                            [-1, 0, -1]]).flatten())


def test_get_orientations():
    board_2d = np.array([[1,  0,  0],
                         [1, -1,  1],
                         [0,  0, -1]])

    board_rot90 = np.array([[0,  1, -1],
                            [0, -1,  0],
                            [1,  1,  0]])

    board_rot180 = np.array([[-1, 0,  0],
                             [1, -1,  1],
                             [0,  0,  1]])

    board_rot270 = np.array([[0,  1,  1],
                             [0, -1,  0],
                             [-1, 1,  0]])

    board_flip_vertical = np.array([[0,  0, -1],
                                    [1, -1,  1],
                                    [1,  0,  0]])

    board_flip_horizontal = np.array([[0,  0,  1],
                                      [1, -1,  1],
                                      [-1, 0,  0]])

    board_rot90_flip_vertical = np.array([[1,  1,  0],
                                          [0, -1,  0],
                                          [0,  1, -1]])

    board_rot90_flip_horizontal = np.array([[-1, 1, 0],
                                            [0, -1, 0],
                                            [0,  1, 1]])

    expected_orientations = [board_2d, board_rot90, board_rot180, board_rot270,
                             board_flip_vertical, board_flip_horizontal,
                             board_rot90_flip_vertical,
                             board_rot90_flip_horizontal]

    orientations = [board_and_transform[0] for board_and_transform
                    in get_symmetrical_board_orientations(board_2d)]

    assert np.array_equal(orientations, expected_orientations)


def test_get_position_value_from_cache():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [0,  0, -1]]).flatten()

    value, found = cache.get_for_position(Board(b))

    assert (value, found) == (None, False)

    cache.set_for_position(Board(b), -1)

    (value, _), found = cache.get_for_position(Board(b))

    assert (value, found) == (-1, True)
