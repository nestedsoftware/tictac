import torch
from torch.nn import MSELoss

from tictac.board import play_random_move
from tictac.qneural import (net_context, create_qneural_player,
                            play_training_games_x,
                            play_training_games_o )


print("Training qlearning X vs. random...")
play_training_games_x(net_context=net_context,
                      o_strategies=[play_random_move])
print("Training qlearning O vs. random...")
play_training_games_o(net_context=net_context,
                      x_strategies=[play_random_move])

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
