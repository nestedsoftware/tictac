#!/usr/bin/env python3
''' Courtesy of NestedSoftware '''


from sys import argv

from numpy import array
from torch import no_grad
from torch.optim import SGD
from torch.nn import MSELoss

from board import play_random_move, play_games, Board
from minimax import create_minimax_player
from qneural import (TicTacNet, NetContext, create_qneural_player, get_q_values,
                     play_training_games_x, play_training_games_o)


GAME_CNT = abs(int(argv[1])) if len(argv) > 1 else 1000

play_minimax_move_randomized = create_minimax_player(True)
play_minimax_move_not_randomized = create_minimax_player(False)

policy_net = TicTacNet()
target_net = TicTacNet()
net_context = NetContext(policy_net, target_net, SGD(policy_net.parameters(), lr=0.1), MSELoss())

with no_grad():
    board = Board(array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"Before training q_values = {q_values}")

print("Training qlearning X vs. random...")
play_training_games_x(net_context=net_context,
                      o_strategies=[play_random_move])
print("Training qlearning O vs. random...")
play_training_games_o(net_context=net_context,
                      x_strategies=[play_random_move])

with no_grad():
    play_qneural_move = create_qneural_player(net_context)
    print("Playing qneural vs random:")
    play_games(GAME_CNT, play_qneural_move, play_random_move)
    print("Playing qneural vs minimax random:")
    play_games(GAME_CNT, play_qneural_move, play_minimax_move_randomized)
    print("Playing qneural vs minimax:")
    play_games(GAME_CNT, play_qneural_move, play_minimax_move_not_randomized)
    print("Playing random vs qneural:")
    play_games(GAME_CNT, play_random_move, play_qneural_move)
    print("Playing minimax random vs qneural:")
    play_games(GAME_CNT, play_minimax_move_randomized, play_qneural_move)
    print("Playing minimax vs qneural:")
    play_games(GAME_CNT, play_minimax_move_not_randomized, play_qneural_move)
    print("Playing qneural vs qneural:")
    play_games(GAME_CNT, play_qneural_move, play_qneural_move)
    board = Board(array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"After training q_values = {q_values}")
