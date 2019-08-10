import numpy as np
import statistics as stats
import random
import operator
import itertools
from collections import deque

from tictac.board import BoardCache, Board
from tictac.board import play_game, play_random_move, is_draw
from tictac.board import (BOARD_SIZE, BOARD_DIMENSIONS, CELL_X, CELL_O,
                          RESULT_X_WINS, RESULT_O_WINS)

# Contains old version of code that required transforming/reverse-transforming
# qvalues. The new version stores the board position after each action as a
# separate key associated with a single key value instead.

WIN_VALUE = 1.0
DRAW_VALUE = 1.0
LOSS_VALUE = -1.0

INITIAL_Q_VALUES_FOR_X = 0.0
INITIAL_Q_VALUES_FOR_O = 0.0


class QTable:
    def __init__(self):
        self.qtable = BoardCache()

    def get_q_values(self, board):
        result, found = self.qtable.get_for_position(board)
        if found:
            qvalues, transform = result
            return reverse_transform_qvalues(qvalues, transform)

        valid_move_indexes = board.get_valid_move_indexes()
        initial_q_value = get_initial_q_value(board)
        initial_q_values = [initial_q_value for _ in valid_move_indexes]

        qvalues = dict(zip(valid_move_indexes, initial_q_values))

        self.qtable.set_for_position(board, qvalues)

        return qvalues

    def get_q_value(self, board, move_index):
        return self.get_q_values(board)[move_index]

    def update_q_value(self, board, move_index, qvalue):
        qvalues = self.get_q_values(board)
        qvalues[move_index] = qvalue

        result, found = self.qtable.get_for_position(board)
        assert found is True, "position must be cached at this point"
        _, transform = result

        transformed_board, transformed_qvalues = transform_board_and_qvalues(
            board, qvalues, transform)

        self.qtable.set_for_position(transformed_board, transformed_qvalues)

    def get_move_index_and_max_q_value(self, board):
        q_values = self.get_q_values(board)
        return max(q_values.items(), key=operator.itemgetter(1))


def get_initial_q_value(board):
    return (INITIAL_Q_VALUES_FOR_X if board.get_turn() == CELL_X
            else INITIAL_Q_VALUES_FOR_O)


def transform_board_and_qvalues(board, q_values, transform):
    b_2d_transformed = transform.transform(board.board_2d)

    q_2d = load_q_values_into_2d_board(q_values)
    q_2d_transformed = transform.transform(q_2d)
    q_values_transformed = dict([(mi, qv) for (mi, qv)
                                 in enumerate(q_2d_transformed.flatten())
                                 if not np.isnan(qv)])

    return Board(b_2d_transformed.flatten()), q_values_transformed


def reverse_transform_qvalues(qvalues, transform):
    qvalues_2d = load_q_values_into_2d_board(qvalues)

    qvalues_2d_transform_reversed = transform.reverse(qvalues_2d)

    return dict([(index, qvalue) for index, qvalue
                 in enumerate(qvalues_2d_transform_reversed.flatten())
                 if not np.isnan(qvalue)])


def load_q_values_into_2d_board(qvalues):
    b = np.empty(BOARD_SIZE**2)
    b[:] = np.nan
    for move_index, qvalue in qvalues.items():
        b[move_index] = qvalue

    return b.reshape(BOARD_DIMENSIONS)


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
    for game in range(total_games):
        move_history = deque()
        strategies = get_strategies_to_use(q_tables, move_history,
                                           x_strategies, o_strategies, epsilon)

        x_strategies_to_use, o_strategies_to_use = strategies

        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(q_tables, move_history, q_table_player,
                           x_strategy_to_use, o_strategy_to_use, learning_rate,
                           discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")


def get_strategies_to_use(q_tables,  move_history, x_strategies, o_strategies,
                          epsilon):
    x_strategies = get_strategies(x_strategies, q_tables, move_history, epsilon)
    o_strategies = get_strategies(o_strategies, q_tables, move_history, epsilon)
    x_strategies_to_use = itertools.cycle(x_strategies)
    o_strategies_to_use = itertools.cycle(o_strategies)
    return x_strategies_to_use, o_strategies_to_use


def get_strategies(strategies, q_tables, move_history, epsilon):
    return ([create_training_player(q_tables, move_history, epsilon)]
            if strategies is None else strategies)


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
        new_q_value = (((1 - learning_rate) * current_q_value)
                       + (learning_rate * discount_factor * game_result_reward))
        q_table.update_q_value(next_position, move_index, new_q_value)

    for (position, move_index) in list(move_history)[1:]:
        current_q_table, next_q_table = get_shuffled_q_tables(q_tables)

        max_next_move_index, _ = current_q_table.get_move_index_and_max_q_value(
            next_position)

        max_next_q_value = next_q_table.get_q_value(next_position,
                                                    max_next_move_index)

        current_q_value = current_q_table.get_q_value(position, move_index)
        new_q_value = (((1 - learning_rate) * current_q_value)
                       + (learning_rate * discount_factor * max_next_q_value))
        current_q_table.update_q_value(position, move_index, new_q_value)

        next_position = position


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
