from unittest import TestCase

from numpy import array, array_equal

from src.board import Board, TRANSFORMATIONS


class TestRandom(TestCase):
    def test_get_random_valid_move(self):
        b = array([0, -1, 0, 0, -1, 0, 1, 0, 1])
        self.assertTrue(Board(b).get_random_valid_move_index() in [0, 2, 3, 5, 7])

class TestTransform(TestCase):
    def test_transform(self):
        b = array([[1,  1, -1],
                   [-1, 1,  1],
                   [1, -1, -1]])
        transformed_b = TRANSFORMATIONS[4](TRANSFORMATIONS[2](b))
        self.assertTrue(array_equal(transformed_b, array([[-1,  1,  1],
                                                          [1,   1, -1],
                                                          [-1, -1,  1]])))
