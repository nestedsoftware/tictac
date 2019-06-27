import itertools
import random
from collections import deque

import numpy as np
from tictac.board_cache import BoardCache
from tictac.common import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS,
                           RESULT_DRAW)
from tictac.common import (play_game, get_turn, play_move, get_game_result,
                           get_valid_move_indexes)
from tictac.minimax import create_minimax_player

qtable = BoardCache()

WIN_VALUE = 1.0
DRAW_VALUE = 0.5
LOSS_VALUE = 0.0

INITIAL_Q_VALUES_FOR_X = 0.01
INITIAL_Q_VALUES_FOR_O = 0.01

play_minimax_move_not_randomized = create_minimax_player(False)


def play_q_table_move(board, q_table=qtable):
    move_index = choose_move(q_table, board, 0)
    return play_move(board, move_index)


def play_training_games_x(total_games=10000, q_table=qtable,
                          learning_rate=0.9, discount_factor=1.0, epsilon=0,
                          o_strategies=[play_minimax_move_not_randomized]):
    play_training_games(total_games, q_table, CELL_X, learning_rate,
                        discount_factor, epsilon, None, o_strategies)


def play_training_games_o(total_games=10000, q_table=qtable,
                          learning_rate=0.1, discount_factor=1.0, epsilon=0,
                          x_strategies=[play_minimax_move_not_randomized]):
    play_training_games(total_games, q_table, CELL_O, learning_rate,
                        discount_factor, epsilon, x_strategies, None)


def play_training_games(total_games, q_table, q_table_player, learning_rate,
                        discount_factor, epsilon, x_strategies, o_strategies):
    for game in range(total_games):
        move_history = deque()
        strategies = get_strategies_to_use(q_table, move_history,
                                           x_strategies, o_strategies,
                                           epsilon)

        x_strategy_to_use = next(strategies[0])
        o_strategy_to_use = next(strategies[1])

        play_training_game(q_table, move_history, q_table_player,
                           x_strategy_to_use, o_strategy_to_use,
                           learning_rate, discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"played {game+1} games, using epsilon={epsilon}...")


def get_strategies_to_use(q_table,  move_history, x_strategies, o_strategies,
                          epsilon):
    x_strategies = get_strategies(x_strategies, q_table, move_history,
                                  epsilon)
    o_strategies = get_strategies(o_strategies, q_table, move_history,
                                  epsilon)
    x_strategies_to_use = itertools.cycle(x_strategies)
    o_strategies_to_use = itertools.cycle(o_strategies)
    return x_strategies_to_use, o_strategies_to_use


def get_strategies(strategy, q_table, move_history, epsilon):
    return ([create_play_for_training(q_table, move_history, epsilon)]
            if strategy is None else strategy)


def play_training_game(q_table, move_history, q_table_player, x_strategy,
                       o_strategy, learning_rate, discount_factor):
    board = play_game(x_strategy, o_strategy)

    update_training_gameover(q_table, move_history, q_table_player, board,
                             learning_rate, discount_factor)


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
        new_q_value = ((1 - learning_rate) * q_value
                       + learning_rate * discount_factor * max_q_value)
        set_q_value(q_table, position, move_index, new_q_value)

        updated_q_values = get_q_values(q_table, position)
        max_q_value = max(updated_q_values)


def create_play_for_training(q_table, move_history, epsilon):
    def play(board):
        move_index = choose_move(q_table, board, epsilon)
        move_history.appendleft((board, move_index))
        return play_move(board, move_index)

    return play


def choose_move(q_table, board, epsilon):
    q_values = get_q_values(q_table, board)
    action_index = choose_action_index(q_values, epsilon)
    valid_move_indexes = get_valid_move_indexes(board)

    return valid_move_indexes[action_index]


def choose_action_index(q_values, epsilon):
    random_value_from_0_to_1 = np.random.uniform()
    if random_value_from_0_to_1 < epsilon:
        return random.randrange(0, len(q_values))

    max_q_value_index = np.argmax(q_values)
    return max_q_value_index


def set_q_value(q_table, board, move_index, q_value):
    q_values, found = q_table.get_for_position(board)
    assert found, "position should already be cached"
    q_table_index = get_valid_move_indexes(board).index(move_index)

    q_values[q_table_index] = q_value
    q_table.set_for_position(board, q_values)


def get_q_values(q_table, board):
    q_values, found = q_table.get_for_position(board)
    if not found:
        initial_q_values = (INITIAL_Q_VALUES_FOR_X
                            if get_turn(board) == CELL_X
                            else INITIAL_Q_VALUES_FOR_O)

        valid_move_indexes = get_valid_move_indexes(board)
        q_values = np.full(len(valid_move_indexes), initial_q_values)
        q_table.set_for_position(board, q_values)

    return q_values


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
