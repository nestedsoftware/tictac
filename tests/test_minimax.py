import pytest

import numpy as np

from tictac import common, minimax


@pytest.fixture(autouse=True)
def reset_cache():
    minimax.reset_cache()


def test_get_position_value_x_wins():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_X_WINS


def test_get_position_value_o_wins():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [0, 1, -1]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_O_WINS


def test_get_position_value_draw():
    board = np.array([[ 1, -1,  1],
                      [ 1,  1, -1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_DRAW


def test_get_position_value_draw_is_best_case():
    board = np.array([[ 1, -1,  0],
                      [ 1,  1, -1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_DRAW


def test_get_position_value_o_wins_in_best_case_x_turn():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  0, -1]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_O_WINS


def test_get_position_value_o_wins_in_best_case_o_turn():
    board = np.array([[1,  0,  0],
                      [1, -1,  1],
                      [0,  0, -1]]).reshape(1, 9)[0]

    value = minimax.get_position_value(board)

    assert value == common.RESULT_O_WINS


def test_get_move_value_pairs_for_position_o_wins_in_best_case():
    board = np.array([[1,  0,  0],
                      [1, -1,  1],
                      [0,  0, -1]]).reshape(1, 9)[0]

    move_value_pairs = minimax.get_move_value_pairs(board)

    assert move_value_pairs == [(1, 1), (2, 1), (6, -1), (7, 1)]


def test_play_minimax_move_o_wins_in_best_case():
    board = np.array([[1,  0,  0],
                      [1, -1,  1],
                      [0,  0, -1]]).reshape(1, 9)[0]

    board = minimax.play_minimax_move(board)

    assert np.array_equal(board, np.array([[ 1,  0,  0],
                                           [ 1, -1,  1],
                                           [-1,  0, -1]]).reshape(1, 9)[0])


def test_get_orientations():
    board_2d = np.array([[1,  0,  0],
                         [1, -1,  1],
                         [0,  0, -1]])

    orientations = minimax.get_symmetrical_board_orientations(board_2d)

    board_rot90 = np.array([[0,  1, -1],
                            [0, -1,  0],
                            [1,  1,  0]])

    board_rot180 = np.array([[-1,  0,  0],
                             [ 1, -1,  1],
                             [ 0,  0,  1]])

    board_rot270 = np.array([[ 0,  1,  1],
                             [ 0, -1,  0],
                             [-1,  1,  0]])

    board_flip_vertical = np.array([[0,  0, -1],
                                    [1, -1,  1],
                                    [1,  0,  0]])

    board_flip_horizontal = np.array([[ 0,  0,  1],
                                      [ 1, -1,  1],
                                      [-1,  0,  0]])

    board_rot90_flip_vertical = np.array([[1,  1,  0],
                                          [0, -1,  0],
                                          [0,  1, -1]])

    board_rot90_flip_horizontal = np.array([[-1,  1, 0],
                                            [ 0, -1, 0],
                                            [ 0,  1, 1]])

    expected_orientations = [board_2d, board_rot90, board_rot180, board_rot270,
                             board_flip_vertical, board_flip_horizontal,
                             board_rot90_flip_vertical,
                             board_rot90_flip_horizontal]

    assert np.array_equal(orientations, expected_orientations)


def test_get_position_value_from_cache():
    board = np.array([[1,  0,  0],
                      [1, -1,  1],
                      [0,  0, -1]]).reshape(1, 9)[0]

    value, found = minimax.get_position_value_from_cache(board)

    assert (value, found) == (None, False)

    minimax.put_position_value_in_cache(board, -1)

    value, found = minimax.get_position_value_from_cache(board)

    assert (value, found) == (-1, True)
