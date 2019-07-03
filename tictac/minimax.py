import random

from tictac.board import BoardCache
from tictac.board import CELL_O
from tictac.board import is_empty

cache = BoardCache()


def create_minimax_player(randomize):
    def play(board):
        return play_minimax_move(board, randomize)

    return play


def play_minimax_move(board, randomize=False):
    move_value_pairs = get_move_value_pairs(board)
    move = filter_best_move(board, move_value_pairs, randomize)

    return board.play_move(move)


def get_move_value_pairs(board):
    valid_move_indexes = board.get_valid_move_indexes()

    assert not is_empty(valid_move_indexes), "never call with an end position"

    move_value_pairs = [(m, get_position_value(board.play_move(m)))
                        for m in valid_move_indexes]

    return move_value_pairs


def get_position_value(board):
    result, found = cache.get_for_position(board)
    if found:
        return result[0]

    position_value = calculate_position_value(board)

    cache.set_for_position(board, position_value)

    return position_value


def calculate_position_value(board):
    if board.is_gameover():
        return board.get_game_result()

    valid_move_indexes = board.get_valid_move_indexes()

    values = [get_position_value(board.play_move(m))
              for m in valid_move_indexes]

    min_or_max = choose_min_or_max_for_comparison(board)
    position_value = min_or_max(values)

    return position_value


def filter_best_move(board, move_value_pairs, randomize):
    min_or_max = choose_min_or_max_for_comparison(board)
    move, value = min_or_max(move_value_pairs, key=lambda mvp: mvp[1])
    if not randomize:
        return move

    best_move_value_pairs = [mvp for mvp in move_value_pairs
                             if mvp[1] == value]
    chosen_move, _ = random.choice(best_move_value_pairs)
    return chosen_move


def choose_min_or_max_for_comparison(board):
    turn = board.get_turn()
    return min if turn == CELL_O else max
