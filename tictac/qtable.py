from collections import deque

import itertools

import numpy as np

from tictac.board_cache import BoardCache

from tictac.common import (play_move, is_gameover, get_game_result,
                           get_valid_move_indexes)
from tictac.common import (CELL_EMPTY, BOARD_SIZE, CELL_X, CELL_O,
                           RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW)

from tictac.random import play_random_move

q_table = BoardCache()

WIN_VALUE = 1.0
DRAW_VALUE = 0.5
LOSS_VALUE = 0.0


def play_q_table_move(board, q_table=q_table):
    action_index = get_action_index(q_table, board, 0)
    move_index = get_valid_move_indexes(board)[action_index]
    return play_move(board, move_index)


def play_training_games(total_games=500, q_table=q_table,
                        q_table_player=CELL_X, history=deque(),
                        learning_rate=0.9, discount_factor=0.9,
                        noise_factor=0.0, x_strategy=None,
                        o_strategy=play_random_move):
    if x_strategy is None:
        x_strategy = create_play_for_training(q_table, history, noise_factor)

    for game in range(total_games):
        play_training_game(q_table, history, q_table_player, x_strategy,
                           o_strategy, learning_rate, discount_factor)


def play_training_game(q_table, history, q_table_player, x_strategy,
                       o_strategy, learning_rate, discount_factor):
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    board = np.array([CELL_EMPTY] * BOARD_SIZE**2)
    while not is_gameover(board):
        play = next(player_strategies)
        board = play(board)

    update_training_gameover(q_table, history, q_table_player, board,
                             learning_rate, discount_factor)


def create_play_for_training(q_table, history, noise_factor):
    def play(board):
        move = get_action_index(q_table, board, noise_factor)
        history.appendleft((board, move))
        return play_move(board, move)

    return play


def update_training_gameover(q_table, history, q_table_player, board,
                             learning_rate, discount_factor):
    new_q_value = get_game_result_value(q_table_player, board)
    last_position, last_move = history[0]
    set_q_value(q_table, last_position, last_move, new_q_value)

    for (position, move) in list(history)[1:]:
        q_value = get_q_values(q_table, position)[move]
        print(q_value)
        new_q_value = ((1-learning_rate) * q_value
                       + learning_rate * (discount_factor * new_q_value))
        set_q_value(q_table, position, move, new_q_value)


def get_game_result_value(player, board):
    if is_win(player, board):
        return WIN_VALUE
    if is_loss(player, board):
        return LOSS_VALUE
    if is_draw(board):
        return DRAW_VALUE


def is_win(player, board):
    result = get_game_result(board)
    return ((player == CELL_O and result == RESULT_O_WINS)
            or (player == CELL_X and result == RESULT_X_WINS))


def is_loss(player, board):
    result = get_game_result(board)
    return ((player == CELL_O and result == RESULT_X_WINS)
            or (player == CELL_X and result == RESULT_O_WINS))


def is_draw(board):
    return get_game_result(board) == RESULT_DRAW


def get_action_index(q_table, board, noise_factor):
    q_values = get_q_values(q_table, board)
    noisy_q_values = get_noisy_q_values(q_values, noise_factor)

    action_index = np.argmax(noisy_q_values)

    return action_index


def set_q_value(q_table, board, action_index, result):
    q_values, found = q_table.get_for_position(board)
    q_values[action_index] = result
    q_table.set_for_position(board, q_values)


def get_q_values(q_table, board):
    q_values, found = q_table.get_for_position(board)
    if not found:
        valid_move_indexes = get_valid_move_indexes(board)
        q_values = np.full(len(valid_move_indexes), 0.5)
        q_table.set_for_position(board, q_values)

    return q_values


def get_noisy_q_values(q_values, noise_rate):
    noise = np.random.randn(len(q_values)) * noise_rate

    return q_values + noise
