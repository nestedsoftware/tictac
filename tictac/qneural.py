from random import randrange

import numpy as np
import itertools
from collections import deque

import torch
from torch import nn

from tictac.board import play_game, is_draw
from tictac.board import (CELL_X, CELL_O, RESULT_X_WINS, RESULT_O_WINS)

WIN_VALUE = 1.0
DRAW_VALUE = 1.0
LOSS_VALUE = -1.0

INITIAL_Q_VALUES_FOR_X = 0.0
INITIAL_Q_VALUES_FOR_O = 0.0

INPUT_SIZE = 9
OUTPUT_SIZE = 9


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_layer = nn.Linear(INPUT_SIZE, 100)
        self.output_layer = nn.Linear(100, OUTPUT_SIZE)

    def forward(self, x):
        x = torch.sigmoid(self.hidden_layer(x))
        x = torch.sigmoid(self.output_layer(x))
        return x


class NetContext:
    def __init__(self, net, optimizer, loss_function):
        self.net = net
        self.optimizer = optimizer
        self.loss_function = loss_function


def create_qneural_player(net_context):
    def play(board):
        return play_qneural_move(board, net_context)

    return play


def play_qneural_move(board, net_context):
    valid_move_indexes = board.get_valid_move_indexes()
    q_values = get_q_values(net_context, board)
    qvalue_move_index_pairs = list(zip(q_values, [i for i in range(9)]))

    valid_q_values = [qvalue_move_index_pairs[mi] for mi in valid_move_indexes]
    max_q_value, max_move_index = max(valid_q_values, key=lambda pair: pair[0])
    return board.play_move(max_move_index)


def get_q_values(net_context, board):
    inputs = convert_to_tensor(board)
    outputs = net_context.net(inputs)
    return outputs


def convert_to_tensor(board):
    return torch.tensor(board.board, dtype=torch.float)


def choose_move_index(net_context, board, epsilon):
    if epsilon > 0:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            next_move_index = randrange(9)
            return next_move_index

    q_values = get_q_values(net_context, board)
    next_move_index = torch.argmax(q_values).item()
    return next_move_index


def play_training_games_x(net_context, total_games=7000, discount_factor=1.0, epsilon=0.7, o_strategies=None):
    play_training_games(net_context, CELL_X, total_games, discount_factor, epsilon, None, o_strategies)


def play_training_games_o(net_context, total_games=7000, discount_factor=1.0, epsilon=0.7, x_strategies=None):
    play_training_games(net_context, CELL_O, total_games, discount_factor, epsilon, x_strategies, None)


def play_training_games(net_context, qplayer, total_games, discount_factor, epsilon, x_strategies, o_strategies):
    for game in range(total_games):
        move_history = deque()
        strategies = get_strategies_to_use(net_context, move_history,
                                           x_strategies, o_strategies, epsilon)

        x_strategies_to_use, o_strategies_to_use = strategies

        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(net_context, move_history, qplayer,
                           x_strategy_to_use, o_strategy_to_use,
                           discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")


def get_strategies_to_use(net_context,  move_history, x_strategies,
                          o_strategies, epsilon):
    x_strategies = get_strategies(x_strategies, net_context, move_history, epsilon)
    o_strategies = get_strategies(o_strategies, net_context, move_history, epsilon)
    x_strategies_to_use = itertools.cycle(x_strategies)
    o_strategies_to_use = itertools.cycle(o_strategies)
    return x_strategies_to_use, o_strategies_to_use


def get_strategies(strategies, net_context, move_history, epsilon):
    return ([create_training_player(net_context, move_history, epsilon)]
            if strategies is None else strategies)


def play_training_game(net_context, move_history, q_learning_player, x_strategy,
                       o_strategy, discount_factor):
    board = play_game(x_strategy, o_strategy)

    update_training_gameover(net_context, move_history, q_learning_player,
                             board, discount_factor)


def update_training_gameover(net_context, move_history, q_learning_player,
                             board, discount_factor):
    game_result_reward = get_game_result_value(q_learning_player, board)

    # move history is in reverse-chronological order - last to first
    next_position, move_index = move_history[0]
    next_output = net_context.net(convert_to_tensor(next_position))
    target = next_output.clone().detach()
    target[move_index] = discount_factor * game_result_reward

    net = net_context.net.train()
    loss = net_context.loss_function(next_output, target)
    net_context.optimizer.zero_grad()
    loss.backward()
    net_context.optimizer.step()

    for (position, move_index) in list(move_history)[1:]:
        output = net(convert_to_tensor(position))

        target = output.clone().detach()
        next_output = net(convert_to_tensor(next_position))
        target[move_index] = torch.max(next_output).item()

        net = net.train()
        loss = net_context.loss_function(output, target)
        net_context.optimizer.zero_grad()
        loss.backward()
        net_context.optimizer.step()

        next_position = position


def create_training_player(net, move_history, epsilon):
    def play(board):
        move_index = choose_move_index(net, board, epsilon)
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
