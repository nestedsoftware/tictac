import math
import numpy as np

from tictac.board import BoardCache, Board
from tictac.mcts import (perform_game_playout, find_or_create_node,
                         calculate_values, perform_training_playouts)


def test_play_mcts_move():
    b_2d = np.array([[1,  1,  0],
                     [1, -1,  0],
                     [-1, 1, -1]])
    b = b_2d.flatten()
    board = Board(b)
    nc = BoardCache()

    parent_node = find_or_create_node(nc, board)
    actual_stats = (parent_node.visits, parent_node.wins, parent_node.draws, parent_node.losses)
    assert actual_stats == (0, 0, 0, 0)

    values = calculate_values(nc, board)
    expected_values = [(2, math.inf), (5, math.inf)]
    assert list(values) == expected_values

    perform_game_playout(nc, board)

    actual_stats = (parent_node.visits, parent_node.wins, parent_node.draws, parent_node.losses)
    assert actual_stats == (1, 0, 0, 1)

    child_node_2 = find_or_create_node(nc, board.play_move(2))
    actual_stats = (child_node_2.visits, child_node_2.wins, child_node_2.draws, child_node_2.losses)
    assert actual_stats == (1, 1, 0, 0)

    child_node_5 = find_or_create_node(nc, board.play_move(5))
    actual_stats = (child_node_5.visits, child_node_5.wins, child_node_5.draws, child_node_5.losses)
    assert actual_stats == (0, 0, 0, 0)

    values = calculate_values(nc, board)
    expected_values = [(2, 1.0), (5, math.inf)]
    assert list(values) == expected_values

    perform_game_playout(nc, board)

    actual_stats = (parent_node.visits, parent_node.wins, parent_node.draws, parent_node.losses)
    assert actual_stats == (2, 1, 0, 1)

    actual_stats = (child_node_2.visits, child_node_2.wins, child_node_2.draws, child_node_2.losses)
    assert actual_stats == (1, 1, 0, 0)

    actual_stats = (child_node_5.visits, child_node_5.wins, child_node_5.draws, child_node_5.losses)
    assert actual_stats == (1, 0, 0, 1)

    values = calculate_values(nc, board)
    expected_values = [(2, 2.177410022515475), (5, 1.1774100225154747)]
    assert list(values) == expected_values

    perform_training_playouts(nc, board, 100, False)

    actual_stats = (parent_node.visits, parent_node.wins, parent_node.draws, parent_node.losses)
    assert actual_stats == (102, 6, 0, 96)

    actual_stats = (child_node_2.visits, child_node_2.wins, child_node_2.draws, child_node_2.losses)
    assert actual_stats == (96, 96, 0, 0)

    actual_stats = (child_node_5.visits, child_node_5.wins, child_node_5.draws, child_node_5.losses)
    assert actual_stats == (6, 0, 0, 6)

    values = calculate_values(nc, board)
    expected_values = [(2, 1.3104087632087014), (5, 1.2416350528348057)]
    assert list(values) == expected_values
