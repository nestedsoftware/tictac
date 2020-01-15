import numpy as np
import statistics as stats
import random
import operator
import itertools
from collections import deque

from tictac.board import BoardCache, Board
from tictac.board import play_game, play_random_move, is_draw
from tictac.board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS)

WIN_VALUE = 1.0
DRAW_VALUE = 1.0
LOSS_VALUE = 0.0

INITIAL_Q_VALUES_FOR_X = 0.0
INITIAL_Q_VALUES_FOR_O = 0.0


class QTable:
    def __init__(self):
        self.qtable = BoardCache()

    def get_q_values(self, board):
        move_indexes = board.get_valid_move_indexes()
        qvalues = [self.get_q_value(board, mi) for mi
                   in board.get_valid_move_indexes()]

        return dict(zip(move_indexes, qvalues))

    def get_q_value(self, board, move_index):
        new_position = board.play_move(move_index)
        result, found = self.qtable.get_for_position(new_position)
        if found is True:
            qvalue, _ = result
            return qvalue

        return get_initial_q_value(new_position)

    def update_q_value(self, board, move_index, qvalue):
        new_position = board.play_move(move_index)

        result, found = self.qtable.get_for_position(new_position)
        if found is True:
            _, t = result
            new_position_transformed = Board(
                t.transform(new_position.board_2d).flatten())
            self.qtable.set_for_position(new_position_transformed, qvalue)
            return

        self.qtable.set_for_position(new_position, qvalue)

    def get_move_index_and_max_q_value(self, board):
        q_values = self.get_q_values(board)
        return max(q_values.items(), key=operator.itemgetter(1))

    def print_q_values(self):
        print(f"num q_values = {len(self.qtable.cache)}")
        for k, v in self.qtable.cache.items():
            b = np.frombuffer(k, dtype=int)
            board = Board(b)
            board.print_board()
            print(f"qvalue = {v}")


def get_initial_q_value(board):
    return (INITIAL_Q_VALUES_FOR_X if board.get_turn() == CELL_O
            else INITIAL_Q_VALUES_FOR_O)


qtables = [QTable()]

double_qtables = [QTable(), QTable()]


def create_q_table_player(q_tables):
    def play(board):
        return play_q_table_move(board, q_tables)

    return play


def play_q_table_move(board, q_tables=None):
    if q_tables is None:
        q_tables = qtables

    move_index = choose_move_index(q_tables, board, 0)
    return board.play_move(move_index)


def choose_move_index(q_tables, board, epsilon):
    if epsilon > 0:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            return board.get_random_valid_move_index()

    move_q_value_pairs = get_move_average_q_value_pairs(q_tables, board)

    return max(move_q_value_pairs, key=lambda pair: pair[1])[0]


def get_move_average_q_value_pairs(q_tables, board):
    move_indexes = sorted(q_tables[0].get_q_values(board).keys())

    avg_q_values = [stats.mean(gather_q_values_for_move(q_tables, board, mi))
                    for mi in move_indexes]

    return list(zip(move_indexes, avg_q_values))


def gather_q_values_for_move(q_tables, board, move_index):
    return [q_table.get_q_value(board, move_index) for q_table in q_tables]


def play_training_games_x(total_games=7000, q_tables=None,
                          learning_rate=0.4, discount_factor=1.0, epsilon=0.7,
                          o_strategies=None):
    if q_tables is None:
        q_tables = qtables
    if o_strategies is None:
        o_strategies = [play_random_move]

    play_training_games(total_games, q_tables, CELL_X, learning_rate,
                        discount_factor, epsilon, None, o_strategies)


def play_training_games_o(total_games=7000, q_tables=None,
                          learning_rate=0.4, discount_factor=1.0, epsilon=0.7,
                          x_strategies=None):
    if q_tables is None:
        q_tables = qtables
    if x_strategies is None:
        x_strategies = [play_random_move]

    play_training_games(total_games, q_tables, CELL_O, learning_rate,
                        discount_factor, epsilon, x_strategies, None)


def play_training_games(total_games, q_tables, q_table_player, learning_rate,
                        discount_factor, epsilon, x_strategies, o_strategies):
    if x_strategies:
        x_strategies_to_use = itertools.cycle(x_strategies)

    if o_strategies:
        o_strategies_to_use = itertools.cycle(o_strategies)

    for game in range(total_games):
        move_history = deque()

        if not x_strategies:
            x = [create_training_player(q_tables, move_history, epsilon)]
            x_strategies_to_use = itertools.cycle(x)

        if not o_strategies:
            o = [create_training_player(q_tables, move_history, epsilon)]
            o_strategies_to_use = itertools.cycle(o)

        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(q_tables, move_history, q_table_player,
                           x_strategy_to_use, o_strategy_to_use, learning_rate,
                           discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")


def play_training_game(q_tables, move_history, q_table_player, x_strategy,
                       o_strategy, learning_rate, discount_factor):
    board = play_game(x_strategy, o_strategy)

    update_training_gameover(q_tables, move_history, q_table_player, board,
                             learning_rate, discount_factor)


def update_training_gameover(q_tables, move_history, q_table_player, board,
                             learning_rate, discount_factor):
    game_result_reward = get_game_result_value(q_table_player, board)

    # move history is in reverse-chronological order - last to first
    next_position, move_index = move_history[0]
    for q_table in q_tables:
        current_q_value = q_table.get_q_value(next_position, move_index)
        new_q_value = calculate_new_q_value(current_q_value, game_result_reward,
                                            0.0, learning_rate, discount_factor)

        q_table.update_q_value(next_position, move_index, new_q_value)

    for (position, move_index) in list(move_history)[1:]:
        current_q_table, next_q_table = get_shuffled_q_tables(q_tables)

        max_next_move_index, _ = current_q_table.get_move_index_and_max_q_value(
            next_position)

        max_next_q_value = next_q_table.get_q_value(next_position,
                                                    max_next_move_index)

        current_q_value = current_q_table.get_q_value(position, move_index)
        new_q_value = calculate_new_q_value(current_q_value, 0.0,
                                            max_next_q_value, learning_rate,
                                            discount_factor)
        current_q_table.update_q_value(position, move_index, new_q_value)

        next_position = position


def calculate_new_q_value(current_q_value, reward, max_next_q_value,
                          learning_rate, discount_factor):
    weighted_prior_values = (1 - learning_rate) * current_q_value
    weighted_new_value = (learning_rate
                          * (reward + discount_factor * max_next_q_value))
    return weighted_prior_values + weighted_new_value


def get_shuffled_q_tables(q_tables):
    q_tables_copy = q_tables.copy()
    random.shuffle(q_tables_copy)
    q_tables_cycle = itertools.cycle(q_tables_copy)

    current_q_table = next(q_tables_cycle)
    next_q_table = next(q_tables_cycle)

    return current_q_table, next_q_table


def create_training_player(q_tables, move_history, epsilon):
    def play(board):
        move_index = choose_move_index(q_tables, board, epsilon)
        move_history.appendleft((board, move_index))
        return board.play_move(move_index)

    return play


def get_game_result_value(player, board):
    if is_win(player, board):
        return WIN_VALUE
    if is_loss(player, board):
        return LOSS_VALUE
    if is_draw(board):
        return DRAW_VALUE


def is_win(player, board):
    result = board.get_game_result()
    return ((player == CELL_O and result == RESULT_O_WINS)
            or (player == CELL_X and result == RESULT_X_WINS))


def is_loss(player, board):
    result = board.get_game_result()
    return ((player == CELL_O and result == RESULT_X_WINS)
            or (player == CELL_X and result == RESULT_O_WINS))
