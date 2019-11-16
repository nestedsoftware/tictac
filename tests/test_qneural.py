import numpy as np
import random

import pytest
import torch
from torch.nn import MSELoss

from tictac.board import Board
from tictac.qneural import TicTacNet, NetContext
from tictac.qneural import convert_to_tensor, create_qneural_player


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_convert_to_tensor():
    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).flatten()

    board = Board(b)

    t = torch.tensor(board.board, dtype=torch.float)

    t = convert_to_tensor(board)

    t_expected = torch.tensor([1., 0., 0., 1., -1., 1., -1., 1., -1.])
    assert torch.all(torch.eq(t_expected, t))


def test_play_qneural_move():
    net = TicTacNet()
    target_net = TicTacNet()
    sgd = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
    loss_function = MSELoss()
    net_context = NetContext(net, target_net, sgd, loss_function)

    play = create_qneural_player(net_context)

    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).flatten()

    board = Board(b)

    updated_board = play(board)

    assert np.array_equal(updated_board.board,
                          np.array([1, -1, 0, 1, -1, 1, -1, 1, -1]))
