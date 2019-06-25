from tictac.common import CELL_O, play_games
from tictac.random_moves import play_random_move
from tictac.minimax import play_minimax_move
from tictac.qtable import (play_training_games_x, play_training_games_o,
                           play_q_table_move)

print("Playing random games: ")
print("---------------------")
play_games(1000, play_random_move, play_random_move)
print("")
print("Playing minimax games: ")
print("----------------------")
play_games(1000, play_minimax_move, play_minimax_move)
print("")
print("Playing minimax vs random games: ")
print("--------------------------------")
play_games(1000, play_minimax_move, play_random_move)
print("")
print("Playing random vs minimax games: ")
print("--------------------------------")
play_games(1000, play_random_move, play_minimax_move)
print("")


print("Training qtable X vs. random/minimax...")
play_training_games_x(o_strategy=[play_random_move, play_minimax_move])
print("Training qtable O vs. random/minimax...")
play_training_games_o(x_strategy=[play_random_move, play_minimax_move])


print("Playing qtable vs random:")
print("-------------------------")
play_games(1000, play_q_table_move, play_random_move)
print("")
print("Playing qtable vs minimax:")
print("--------------------------")
play_games(1000, play_q_table_move, play_minimax_move)
print("")
print("Playing qtable vs qtable:")
print("-------------------------")
play_games(1000, play_q_table_move, play_q_table_move)
print("")

print("Playing random vs qtable:")
print("-------------------------")
play_games(1000, play_random_move, play_q_table_move)
print("")
print("Playing minimax vs qtable:")
print("--------------------------")
play_games(1000, play_minimax_move, play_q_table_move)
print("")


