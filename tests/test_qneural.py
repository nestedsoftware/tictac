from random import seed as rand_seed
from unittest import TestCase

from numpy import array, array_equal
from numpy.random import seed as np_seed
from torch import manual_seed, tensor, all as tall, eq as teq, float as tfloat
from torch.optim import SGD
from torch.nn import MSELoss

from src.board import Board
from src.qneural import TicTacNet, NetContext, convert_to_tensor, create_qneural_player


class TestQneural(TestCase):
    def setUp(self):
        rand_seed(0)
        np_seed(0)
        manual_seed(0)

    def test_convert_to_tensor(self):
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [-1, 1, -1]]).flatten()
        board = Board(b)
        t = tensor(board.board, dtype=tfloat)
        t = convert_to_tensor(board)
        t_expected = tensor([1., 0., 0., 1., -1., 1., -1., 1., -1.])
        self.assertTrue(tall(teq(t_expected, t)))

    def test_play_qneural_move(self):
        net = TicTacNet()
        target_net = TicTacNet()
        sgd = SGD(net.parameters(), lr=0.1, weight_decay=0)
        loss_function = MSELoss()
        net_context = NetContext(net, target_net, sgd, loss_function)
        play = create_qneural_player(net_context)
        b = array([[1,  0,  0],
                   [1, -1,  1],
                   [-1, 1, -1]]).flatten()
        board = Board(b)
        updated_board = play(board)
        self.assertTrue(array_equal(updated_board.board, array([1, -1, 0, 1, -1, 1, -1, 1, -1])))
