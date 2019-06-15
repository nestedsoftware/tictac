from tictac.common import play_games
from tictac.random import play_random_move
from tictac.minimax import play_minimax_move

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
