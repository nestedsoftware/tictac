from tictac.common import CELL_O, play_games
from tictac.random_moves import play_random_move
from tictac.minimax import create_minimax_player
from tictac.qtable import (play_training_games_x, play_training_games_o,
                           play_q_table_move)

play_minimax_move_randomized = create_minimax_player(True)
play_minimax_move_not_randomized = create_minimax_player(False)

print("Playing random vs random:")
print("-------------------------")
play_games(1000, play_random_move, play_random_move)
print("")
print("Playing minimax not random vs minimax random:")
print("----------------------------------------------")
play_games(1000, play_minimax_move_not_randomized,
           play_minimax_move_randomized)
print("")
print("Playing minimax random vs random:")
print("--------------------------------")
play_games(1000, play_minimax_move_randomized, play_random_move)
print("")
print("Playing random vs minimax random:")
print("---------------------------------")
play_games(1000, play_random_move, play_minimax_move_randomized)
print("")

print("Training qtable X vs. random and minimax random...")
play_training_games_x(o_strategies=[play_random_move,
                                    play_minimax_move_randomized])
print("Training qtable O vs. random and minimax random...")
play_training_games_o(x_strategies=[play_random_move,
                                    play_minimax_move_randomized])

print("Playing qtable vs random:")
print("-------------------------")
play_games(1000, play_q_table_move, play_random_move)
print("")
print("Playing qtable vs minimax random:")
print("---------------------------------")
play_games(1000, play_q_table_move, play_minimax_move_randomized)
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
play_games(1000, play_minimax_move_randomized, play_q_table_move)
print("")

