import numpy as np

from tictac import common


def test_get_valid_move_indexes():
    board = np.array([0,-1,0,0,-1,0,1,0,1])

    valid_indexes = common.get_valid_move_indexes(board)

    assert valid_indexes == [0,2,3,5,7]


def test_get_rows_cols_and_diagonals():
    board = np.array([[1, 1,-1],
                      [0, 1,-1],
                      [0,-1, 1]])
    rows_cols_and_diagonals = common.get_rows_cols_and_diagonals(board)

    expected_rows_cols_and_diagonals = [
        np.array([1,1,-1]),
        np.array([0,1,-1]),
        np.array([0,-1,1]),
        np.array([1,1,1]),
        np.array([-1,-1,1]),
        np.array([1,1,-1]),
        np.array([1,0,0]),
        np.array([-1,1,0])]

    assert np.array_equal(rows_cols_and_diagonals,
                          expected_rows_cols_and_diagonals)


def test_get_game_result_x_wins():
    board = np.array([[1, 1,-1],
                      [0, 1,-1],
                      [0,-1, 1]]).reshape(1,9)[0]

    result = common.get_game_result(board)

    assert result == common.RESULT_X_WINS


def test_get_game_result_o_wins():
    board = np.array([[ 1, 0,-1],
                      [ 0,-1, 1],
                      [-1, 0, 1]]).reshape(1,9)[0]

    result = common.get_game_result(board)

    assert result == common.RESULT_O_WINS


def test_get_game_result_draw():
    board = np.array([[ 1, 1,-1],
                      [-1,-1, 1],
                      [ 1,-1, 1]]).reshape(1,9)[0]

    result = common.get_game_result(board)

    assert result == common.RESULT_DRAW


def test_get_game_result_not_over():
    board = np.array([[ 1, 1, -1],
                      [ 0,-1, 0],
                      [ 1,-1, 1]]).reshape(1,9)[0]

    result = common.get_game_result(board)

    assert result == common.RESULT_NOT_OVER
