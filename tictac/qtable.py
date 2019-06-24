import itertools
from collections import deque

import numpy as np
from tictac.board_cache import BoardCache
from tictac.common import (CELL_EMPTY, BOARD_SIZE, CELL_X, CELL_O,
                           RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW)
from tictac.common import (play_move, is_gameover, get_game_result,
                           get_valid_move_indexes)
from tictac.minimax import play_minimax_move

qtable = BoardCache()

WIN_VALUE = 1.0
DRAW_VALUE = 0.5
LOSS_VALUE = 0.0


def play_q_table_move(board, q_table=qtable):
    move_index = choose_move(q_table, board, 0)
    return play_move(board, move_index)


def play_training_games(total_games=10000, q_table=qtable,
                        q_table_player=CELL_X, learning_rate=0.9,
                        discount_factor=0.9, noise_factor=0.0,
                        x_strategy=None, o_strategy=play_minimax_move):
    i = 1
    for game in range(total_games):
        move_history = deque()
        x_strategy_set = get_player_strategy(x_strategy, q_table,
                                             move_history, noise_factor)
        o_strategy_set = get_player_strategy(o_strategy, q_table,
                                             move_history, noise_factor)

        play_training_game(q_table, move_history, q_table_player,
                           x_strategy_set, o_strategy_set, learning_rate,
                           discount_factor)
        if i % (total_games/10) == 0:
            print(f"played {i} games...")
        i += 1


def get_player_strategy(player_strategy, q_table, move_history, noise_factor):
    if player_strategy is None:
        return create_play_for_training(q_table, move_history, noise_factor)

    return player_strategy


def play_training_game(q_table, move_history, q_table_player, x_strategy,
                       o_strategy, learning_rate, discount_factor):
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    board = np.array([CELL_EMPTY] * BOARD_SIZE**2)
    while not is_gameover(board):
        play = next(player_strategies)
        board = play(board)

    update_training_gameover(q_table, move_history, q_table_player, board,
                             learning_rate, discount_factor)


def create_play_for_training(q_table, move_history, noise_factor):
    def play(board):
        move_index = choose_move(q_table, board, noise_factor)
        move_history.appendleft((board, move_index))
        return play_move(board, move_index)

    return play


def update_training_gameover(q_table, move_history, q_table_player, board,
                             learning_rate, discount_factor):
    new_q_value = get_game_result_value(q_table_player, board)
    final_position, final_move = move_history[0]
    set_q_value(q_table, final_position, final_move, new_q_value)

    updated_q_values = get_q_values(q_table, final_position)
    max_q_value = max(updated_q_values)

    for (position, move_index) in list(move_history)[1:]:
        q_values = get_q_values(q_table, position)
        q_value_index = get_valid_move_indexes(position).index(move_index)
        q_value = q_values[q_value_index]
        new_q_value = ((1-learning_rate) * q_value
                       + learning_rate * (discount_factor * max_q_value))
        set_q_value(q_table, position, move_index, new_q_value)

        updated_q_values = get_q_values(q_table, position)
        max_q_value = max(updated_q_values)


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


def choose_move(q_table, board, noise_factor):
    q_values = get_q_values(q_table, board)
    noisy_q_values = get_noisy_q_values(q_values, noise_factor)

    action_index = np.argmax(noisy_q_values)

    valid_move_indexes = get_valid_move_indexes(board)

    return valid_move_indexes[action_index]


def set_q_value(q_table, board, move_index, q_value):
    q_values, found = q_table.get_for_position(board)
    assert found, "position should already be cached"
    q_table_index = get_valid_move_indexes(board).index(move_index)

    q_values[q_table_index] = q_value
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
