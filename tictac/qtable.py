import numpy as np

from tictac.common import (get_turn, play_move, is_gameover, get_game_result,
                           get_valid_move_indexes)
from tictac.common import CELL_O


def train_episode(q_table, board, learning_rate, discount_factor,
                  noise_factor):
    action_index = get_action_index(q_table, board, noise_factor)

    move_index = get_valid_move_indexes(board)[action_index]
    next_board = play_move(board, move_index)

    if is_gameover(next_board):
        result = get_game_result(next_board)
        set_q_value(q_table, board, action_index, result)
        return

    train_episode(q_table, next_board, learning_rate, discount_factor,
                  noise_factor)

    current_q_value = get_q_values(q_table, board)[action_index]

    next_q_values = get_q_values(q_table, next_board)
    min_or_max_q_value_from_next_state = get_min_or_max_q_value(board,
                                                                next_q_values)

    new_q_value = ((1 - learning_rate) * current_q_value
                   + (learning_rate * discount_factor
                      * min_or_max_q_value_from_next_state))

    set_q_value(q_table, board, action_index, new_q_value)


def get_action_index(q_table, board, noise_factor):
    q_values = get_q_values(q_table, board)
    noisy_q_values = get_noisy_q_values(q_values, noise_factor)

    argmin_or_argmax = choose_argmin_or_argmax_for_comparison(board)
    action_index = argmin_or_argmax(noisy_q_values)

    return action_index


def set_q_value(q_table, board, action_index, result):
    q_values, found = q_table.get_for_position(board)
    q_values[action_index] = result
    q_table.set_for_position(board, q_values)


def get_q_values(q_table, board):
    q_values, found = q_table.get_for_position(board)
    if not found:
        valid_move_indexes = get_valid_move_indexes(board)
        q_values = np.zeros(len(valid_move_indexes))
        q_table.set_for_position(board, q_values)

    return q_values


def get_min_or_max_q_value(board, q_values):
    argmin_or_argmax = choose_argmin_or_argmax_for_comparison(board)
    min_or_max_q_value_index = argmin_or_argmax(q_values)

    return q_values[min_or_max_q_value_index]


def choose_argmin_or_argmax_for_comparison(board):
    turn = get_turn(board)
    return np.argmin if turn == CELL_O else np.argmax


def get_noisy_q_values(q_values, noise_rate):
    noise = np.random.randn(len(q_values)) * noise_rate

    return q_values + noise
