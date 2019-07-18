import numpy as np

from tictac.board import (RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW,
                          RESULT_NOT_OVER)
from tictac.board import Board, get_rows_cols_and_diagonals


def test_get_valid_move_indexes():
    board = Board(np.array([0, -1, 0, 0, -1, 0, 1, 0, 1]))

    valid_indexes = board.get_valid_move_indexes()

    assert valid_indexes == [0, 2, 3, 5, 7]


def test_get_rows_cols_and_diagonals():
    board = np.array([[1,  1, -1],
                      [0,  1, -1],
                      [0, -1,  1]])
    rows_cols_and_diagonals = get_rows_cols_and_diagonals(board)

    expected_rows_cols_and_diagonals = [
        np.array([1, 1, -1]),
        np.array([0, 1, -1]),
        np.array([0, -1, 1]),
        np.array([1, 1, 1]),
        np.array([-1, -1, 1]),
        np.array([1, 1, -1]),
        np.array([1, 0, 0]),
        np.array([-1, 1, 0])]

    assert np.array_equal(rows_cols_and_diagonals,
                          expected_rows_cols_and_diagonals)


def test_get_game_result_x_wins():
    b = np.array([[1,  1, -1],
                  [0,  1, -1],
                  [0, -1,  1]]).flatten()

    board = Board(b)

    result = board.get_game_result()

    assert result == RESULT_X_WINS


def test_get_game_result_o_wins():
    b = np.array([[1,  0, -1],
                  [0, -1,  1],
                  [-1, 0,  1]]).flatten()

    board = Board(b)

    result = board.get_game_result()

    assert result == RESULT_O_WINS


def test_get_game_result_draw():
    b = np.array([[1,   1, -1],
                  [-1, -1,  1],
                  [1,  -1,  1]]).flatten()

    board = Board(b)

    result = board.get_game_result()

    assert result == RESULT_DRAW


def test_get_game_result_not_over():
    b = np.array([[1,  1, -1],
                  [0, -1,  0],
                  [1, -1,  1]]).flatten()

    board = Board(b)

    result = board.get_game_result()

    assert result == RESULT_NOT_OVER
