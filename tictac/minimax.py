import numpy as np

from tictac.common import CELL_O

from tictac.common import (get_game_result, is_gameover, play_move,
                           get_valid_move_indexes, get_board_2d, get_turn,
                           not_empty)

cache = {}


def reset_cache():
    global cache
    cache = {}


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
    cached_position_value, found = get_position_value_from_cache(board)
    if found:
        return cached_position_value

    position_value = calculate_position_value(board)

    put_position_value_in_cache(board, position_value)

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
    move, value = min_or_max(move_value_pairs, key=lambda mvp: mvp[1])
    return move


def choose_min_or_max_for_comparison(board):
    turn = get_turn(board)
    return min if turn == CELL_O else max


def put_position_value_in_cache(board, value):
    board_2d = get_board_2d(board)
    cache[board_2d.tobytes()] = value


def get_position_value_from_cache(board):
    board_2d = get_board_2d(board)
    board_orientations = get_symmetrical_board_orientations(board_2d)

    for b in board_orientations:
        result = cache.get(b.tobytes(), "not_found")
        if result != "not_found":
            return result, True

    return None, False


def get_symmetrical_board_orientations(board_2d):
    orientations = [board_2d]

    current_board_2d = board_2d
    for i in range(3):
        current_board_2d = np.rot90(current_board_2d)
        orientations.append(current_board_2d)

    orientations.append(np.flipud(board_2d))
    orientations.append(np.fliplr(board_2d))

    orientations.append(np.flipud(np.rot90(board_2d)))
    orientations.append(np.fliplr(np.rot90(board_2d)))

    return orientations
