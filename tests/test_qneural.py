import numpy as np
import random

import pytest
import torch
from torch.nn import MSELoss

from tictac.board import Board
from tictac.qneural import TicTacNet, NetContext, create_qneural_player


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)


def test_play_qneural_move():
    net = TicTacNet()
    sgd = torch.optim.SGD(net.parameters(), lr=0.1, weight_decay=0)
    loss = MSELoss()
    net_context = NetContext(net, sgd, loss)

    play = create_qneural_player(net_context)

    b = np.array([[1,  0,  0],
                  [1, -1,  1],
                  [-1, 1, -1]]).flatten()

    board = Board(b)

    updated_board = play(board)

    assert np.array_equal(updated_board.board,
                          np.array([1, 0, -1, 1, -1, 1, -1, 1, -1]))
