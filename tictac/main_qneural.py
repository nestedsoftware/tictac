import numpy as np

import torch
from torch.nn import MSELoss

from tictac.board import play_random_move, play_games, Board
from tictac.minimax import create_minimax_player
from tictac.qneural import (TicTacNet, NetContext, create_qneural_player,
                            get_q_values, play_training_games_x,
                            play_training_games_o)

play_minimax_move_randomized = create_minimax_player(True)
play_minimax_move_not_randomized = create_minimax_player(False)


policy_net = TicTacNet()
target_net = TicTacNet()
sgd = torch.optim.SGD(policy_net.parameters(), lr=0.1)
loss = MSELoss()
net_context = NetContext(policy_net, target_net, sgd, loss)

with torch.no_grad():
    board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"Before training q_values = {q_values}")

print("Training qlearning X vs. random...")
play_training_games_x(net_context=net_context,
                      o_strategies=[play_random_move])
print("Training qlearning O vs. random...")
play_training_games_o(net_context=net_context,
                      x_strategies=[play_random_move])

print("")

with torch.no_grad():
    play_qneural_move = create_qneural_player(net_context)

    print("Playing qneural vs random:")
    print("--------------------------")
    play_games(1000, play_qneural_move, play_random_move)
    print("")
    print("Playing qneural vs minimax random:")
    print("----------------------------------")
    play_games(1000, play_qneural_move, play_minimax_move_randomized)
    print("")
    print("Playing qneural vs minimax:")
    print("---------------------------")
    play_games(1000, play_qneural_move, play_minimax_move_not_randomized)
    print("")

    print("Playing random vs qneural:")
    print("--------------------------")
    play_games(1000, play_random_move, play_qneural_move)
    print("")
    print("Playing minimax random vs qneural:")
    print("----------------------------------")
    play_games(1000, play_minimax_move_randomized, play_qneural_move)
    print("")
    print("Playing minimax vs qneural:")
    print("---------------------------")
    play_games(1000, play_minimax_move_not_randomized, play_qneural_move)
    print("")

    print("Playing qneural vs qneural:")
    print("---------------------------")
    play_games(1000, play_qneural_move, play_qneural_move)
    print("")

    board = Board(np.array([1, -1, -1, 0, 1, 1, 0, 0, -1]))
    q_values = get_q_values(board, net_context.target_net)
    print(f"After training q_values = {q_values}")
