import numpy as np
import operator
import itertools
from collections import deque

from tictac.board import BoardCache
from tictac.board import play_game
from tictac.board import (BOARD_SIZE, BOARD_DIMENSIONS, CELL_X, CELL_O,
                          RESULT_X_WINS, RESULT_O_WINS, RESULT_DRAW)

from tictac.minimax import create_minimax_player

WIN_VALUE = 1.0
DRAW_VALUE = 0.0
LOSS_VALUE = -1.0

INITIAL_Q_VALUES_FOR_X = 0.0
INITIAL_Q_VALUES_FOR_O = 0.0

play_minimax_move_randomized = create_minimax_player(True)


class QTable:
    def __init__(self):
        self.qtable = BoardCache()

    def get_q_values(self, board):
        result, found = self.qtable.get_for_position(board)
        if found:
            qvalues, t = result
            return get_transformed_move_indexes_and_q_values(qvalues, t)

        valid_move_indexes = board.get_valid_move_indexes()
        initial_q_value = get_initial_q_value(board)
        initial_q_values = [initial_q_value for _ in valid_move_indexes]
        qvalues = dict(zip(valid_move_indexes, initial_q_values))

        self.qtable.set_for_position(board, qvalues)

        return qvalues

    def update_q_value(self, board, move_index, qvalue):
        qvalues = self.get_q_values(board)
        qvalues[move_index] = qvalue
        self.qtable.set_for_position(board, qvalues)

    def get_move_index_and_max_q_value(self, board):
        q_values = self.get_q_values(board)
        return max(q_values.items(), key=operator.itemgetter(1))


def get_initial_q_value(board):
    return (INITIAL_Q_VALUES_FOR_X if board.get_turn() == CELL_X
            else INITIAL_Q_VALUES_FOR_O)


def get_transformed_move_indexes_and_q_values(qvalues, t):
    b_2d = load_q_values_into_2d_board(qvalues)

    reversed_b_2d = t.reverse(b_2d)

    return dict([(index, qvalue) for index, qvalue
                 in enumerate(reversed_b_2d.flatten()) if not np.isnan(qvalue)])


def load_q_values_into_2d_board(qvalues):
    b = np.empty(BOARD_SIZE ** 2)
    b[:] = np.nan
    for move_index, qvalue in qvalues.items():
        b[move_index] = qvalue
    return b.reshape(BOARD_DIMENSIONS)


qtable = QTable()


def play_q_table_move(board, q_table=qtable):
    move = choose_move_index(q_table, board, 0)
    return board.play_move(move)


def choose_move_index(q_table, board, epsilon):
    if epsilon:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            return board.get_random_valid_move_index()

    return q_table.get_move_index_and_max_q_value(board)[0]


def play_training_games_x(total_games=10000, q_table=qtable,
                          learning_rate=0.9, discount_factor=1.0, epsilon=0.8,
                          o_strategies=None):
    if not o_strategies:
        o_strategies = [play_minimax_move_randomized]
    play_training_games(total_games, q_table, CELL_X, learning_rate,
                        discount_factor, epsilon, None, o_strategies)


def play_training_games_o(total_games=10000, q_table=qtable,
                          learning_rate=0.9, discount_factor=1.0, epsilon=0.95,
                          x_strategies=None):
    if not x_strategies:
        x_strategies = [play_minimax_move_randomized]
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
    q_table.update_q_value(final_position, final_move, new_q_value)
    max_q_value = q_table.get_move_index_and_max_q_value(final_position)[1]

    for (position, move_index) in list(move_history)[1:]:
        q_values = q_table.get_q_values(position)
        q_value = q_values[move_index]
        new_q_value = ((1 - learning_rate) * q_value
                       + learning_rate * discount_factor * max_q_value)
        q_table.update_q_value(position, move_index, new_q_value)
        max_q_value = q_table.get_move_index_and_max_q_value(position)[1]


def create_play_for_training(q_table, move_history, epsilon):
    def play(board):
        move_index = choose_move_index(q_table, board, epsilon)
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


def is_draw(board):
    return board.get_game_result() == RESULT_DRAW
