from unittest import TestCase

from numpy import array, array_equal
from src.board import (RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW, RESULT_NOT_OVER,
                       Board, get_rows_cols_and_diagonals)


class TestBoard(TestCase):
    def test_get_valid_move_indexes(self):
        valid_indexes = Board(array([0, -1, 0, 0, -1, 0, 1, 0, 1])).get_valid_move_indexes()
        self.assertEqual(valid_indexes, [0, 2, 3, 5, 7])

    def test_get_rows_cols_and_diagonals(self):
        board = array([[1,  1, -1],
                       [0,  1, -1],
                       [0, -1,  1]])
        rows_cols_and_diagonals = get_rows_cols_and_diagonals(board)
        expected_rows_cols_and_diagonals = [
            array([1, 1, -1]),
            array([0, 1, -1]),
            array([0, -1, 1]),
            array([1, 1, 1]),
            array([-1, -1, 1]),
            array([1, 1, -1]),
            array([1, 0, 0]),
            array([-1, 1, 0])]
        self.assertTrue(array_equal(rows_cols_and_diagonals, expected_rows_cols_and_diagonals))

    def test_get_game_result_x_wins(self):
        b = array([[1,  1, -1],
                   [0,  1, -1],
                   [0, -1,  1]]).flatten()
        self.assertEqual(Board(b).get_game_result(), RESULT_X_WINS)

    def test_get_game_result_o_wins(self):
        b = array([[1,  0, -1],
                   [0, -1,  1],
                   [-1, 0,  1]]).flatten()
        self.assertEqual(Board(b).get_game_result(), RESULT_O_WINS)

    def test_get_game_result_draw(self):
        b = array([[1,   1, -1],
                   [-1, -1,  1],
                   [1,  -1,  1]]).flatten()
        self.assertEqual(Board(b).get_game_result(), RESULT_DRAW)

    def test_get_game_result_not_over(self):
        b = array([[1,  1, -1],
                   [0, -1,  0],
                   [1, -1,  1]]).flatten()
        self.assertEqual(Board(b).get_game_result(), RESULT_NOT_OVER)
