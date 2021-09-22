from unittest import TestCase

from numpy import array, array_equal

from src.minimax import cache, get_position_value, get_move_value_pairs, play_minimax_move
from src.board import (Board, RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW,
                       get_symmetrical_board_orientations)


class TestMinimax(TestCase):
    def setUp(self):
        cache.reset()

    def test_get_position_value_x_wins(self):
        b = array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0,  0]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_X_WINS)

    def test_get_position_value_o_wins(self):
        b = array([[1, 0, -1],
                   [1, 0, -1],
                   [0, 1, -1]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_O_WINS)

    def test_get_position_value_draw(self):
        b = array([[1, -1,  1],
                   [1,  1, -1],
                   [-1, 1, -1]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_DRAW)

    def test_get_position_value_draw_is_best_case(self):
        b = array([[1, -1,  0],
                   [1,  1, -1],
                   [-1, 1, -1]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_DRAW)

    def test_get_position_value_o_wins_in_best_case_x_turn(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [-1, 0, -1]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_O_WINS)

    def test_get_position_value_o_wins_in_best_case_o_turn(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [0,  0, -1]]).flatten()
        self.assertEqual(get_position_value(Board(b)), RESULT_O_WINS)

    def test_get_move_value_pairs_for_position_o_wins_in_best_case(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [0,  0, -1]]).flatten()
        self.assertEqual(get_move_value_pairs(Board(b)), [(1, 1), (2, 1), (6, -1), (7, 1)])

    def test_play_minimax_move_o_wins_in_best_case(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [0,  0, -1]]).flatten()
        result = play_minimax_move(Board(b)).board
        self.assertTrue(array_equal(result, array([[1,  0,  0],
                                                   [1, -1,  1],
                                                   [-1, 0, -1]]).flatten()))

    def test_get_orientations(self):
        board_2d = array([[1,  0,  0],
                          [1, -1,  1],
                          [0,  0, -1]])
        board_rot90 = array([[0,  1, -1],
                             [0, -1,  0],
                             [1,  1,  0]])
        board_rot180 = array([[-1, 0,  0],
                              [1, -1,  1],
                              [0,  0,  1]])
        board_rot270 = array([[0,  1,  1],
                              [0, -1,  0],
                              [-1, 1,  0]])
        board_flip_vertical = array([[0,  0, -1],
                                     [1, -1,  1],
                                     [1,  0,  0]])
        board_flip_horizontal = array([[0,  0,  1],
                                       [1, -1,  1],
                                       [-1, 0,  0]])
        board_rot90_flip_vertical = array([[1,  1,  0],
                                           [0, -1,  0],
                                           [0,  1, -1]])
        board_rot90_flip_horizontal = array([[-1, 1, 0],
                                             [0, -1, 0],
                                             [0,  1, 1]])
        expected_orientations = [board_2d, board_rot90, board_rot180, board_rot270,
                                 board_flip_vertical, board_flip_horizontal,
                                 board_rot90_flip_vertical,
                                 board_rot90_flip_horizontal]
        orientations = [board_and_transform[0] for board_and_transform
                        in get_symmetrical_board_orientations(board_2d)]
        self.assertTrue(array_equal(orientations, expected_orientations))

    def test_get_position_value_from_cache(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [0,  0, -1]]).flatten()
        value, found = cache.get_for_position(Board(b))
        assert (value, found) == (None, False)
        cache.set_for_position(Board(b), -1)
        (value, _), found = cache.get_for_position(Board(b))
        self.assertEqual((value, found), (-1, True))
