import random

from tictac.board_cache import BoardCache
from tictac.common import CELL_O
from tictac.common import (get_game_result, is_gameover, play_move,
                           get_valid_move_indexes, get_turn, not_empty)

cache = BoardCache()


def play_minimax_move(board):
    move_value_pairs = get_move_value_pairs(board)
    move = filter_best_move(board, move_value_pairs)

    return play_move(board, move)


def get_move_value_pairs(board):
    valid_move_indexes = get_valid_move_indexes(board)

    assert not_empty(valid_move_indexes), "never call with an end-position"

    move_value_pairs = [(m, get_position_value(play_move(board, m)))
                        for m in valid_move_indexes]

    return move_value_pairs


def get_position_value(board):
    cached_position_value, found = cache.get_for_position(board)
    if found:
        return cached_position_value

    position_value = calculate_position_value(board)

    cache.set_for_position(board, position_value)

    return position_value


def calculate_position_value(board):
    if is_gameover(board):
        return get_game_result(board)

    valid_move_indexes = get_valid_move_indexes(board)

    values = [get_position_value(play_move(board, m))
              for m in valid_move_indexes]

    min_or_max = choose_min_or_max_for_comparison(board)
    position_value = min_or_max(values)

    return position_value


def filter_best_move(board, move_value_pairs):
    min_or_max = choose_min_or_max_for_comparison(board)

    move, _ = min_or_max(move_value_pairs, key=lambda mvp: mvp[1])

    return move


def choose_min_or_max_for_comparison(board):
    turn = get_turn(board)
    return min if turn == CELL_O else max
