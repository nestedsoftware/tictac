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
LOSS_VALUE = 0.0

INPUT_SIZE = 9
OUTPUT_SIZE = 9


class TicTacNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl1 = nn.Linear(INPUT_SIZE, 36)
        self.dl2 = nn.Linear(36, 36)
        self.output_layer = nn.Linear(36, OUTPUT_SIZE)

    def forward(self, x):
        x = self.dl1(x)
        x = torch.relu(x)

        x = self.dl2(x)
        x = torch.relu(x)

        x = self.output_layer(x)
        x = torch.sigmoid(x)
        return x


class NetContext:
    def __init__(self, policy_net, target_net, optimizer, loss_function):
        self.policy_net = policy_net

        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net = self.target_net.eval()

        self.optimizer = optimizer
        self.loss_function = loss_function


def create_qneural_player(net_context):
    def play(board):
        model = net_context.target_net
        return play_qneural_move(board, model)

    return play


def play_qneural_move(board, model):
    max_move_index, _ = select_valid_qneural_move(board, model)
    return board.play_move(max_move_index)


def select_valid_qneural_move(board, model):
    q_values = get_q_values(board, model)
    valid_move_indexes = board.get_valid_move_indexes()
    valid_q_values = get_valid_move_index_q_value_pairs(q_values,
                                                        valid_move_indexes)
    max_move_index, q_value = max(valid_q_values, key=lambda pair: pair[1])

    return max_move_index, q_value


def get_valid_move_index_q_value_pairs(q_values, valid_move_indexes):
    valid_q_values = []
    for vmi in valid_move_indexes:
        valid_q_values.append((vmi, q_values[vmi].item()))

    return valid_q_values


def get_q_values(board, model):
    inputs = convert_to_tensor(board)
    outputs = model(inputs)
    return outputs


def convert_to_tensor(board):
    return torch.tensor(board.board, dtype=torch.float)


def play_training_games_x(net_context, total_games=2000000,
                          discount_factor=1.0, epsilon=0.7, o_strategies=None):
    play_training_games(net_context, CELL_X, total_games, discount_factor,
                        epsilon, None, o_strategies)


def play_training_games_o(net_context, total_games=2000000,
                          discount_factor=1.0, epsilon=0.7, x_strategies=None):
    play_training_games(net_context, CELL_O, total_games, discount_factor,
                        epsilon, x_strategies, None)


def play_training_games(net_context, qplayer, total_games, discount_factor,
                        epsilon, x_strategies, o_strategies):
    if x_strategies:
        x_strategies_to_use = itertools.cycle(x_strategies)

    if o_strategies:
        o_strategies_to_use = itertools.cycle(o_strategies)

    for game in range(total_games):
        move_history = deque()

        if not x_strategies:
            x = [create_training_player(net_context, move_history, epsilon)]
            x_strategies_to_use = itertools.cycle(x)

        if not o_strategies:
            o = [create_training_player(net_context, move_history, epsilon)]
            o_strategies_to_use = itertools.cycle(o)


        x_strategy_to_use = next(x_strategies_to_use)
        o_strategy_to_use = next(o_strategies_to_use)

        play_training_game(net_context, move_history, qplayer,
                           x_strategy_to_use, o_strategy_to_use,
                           discount_factor)

        if (game+1) % (total_games / 10) == 0:
            epsilon = max(0, epsilon - 0.1)
            print(f"{game+1}/{total_games} games, using epsilon={epsilon}...")


def play_training_game(net_context, move_history, q_learning_player,
                       x_strategy, o_strategy, discount_factor):
    board = play_game(x_strategy, o_strategy)

    update_training_gameover(net_context, move_history, q_learning_player,
                             board, discount_factor)


def update_training_gameover(net_context, move_history, q_learning_player,
                             final_board, discount_factor):
    game_result_reward = get_game_result_value(q_learning_player, final_board)

    # move history is in reverse-chronological order - last to first
    next_position, move_index = move_history[0]

    backpropagate(net_context, next_position, move_index, game_result_reward)

    for (position, move_index) in list(move_history)[1:]:
        with torch.no_grad():
            next_q_values = get_q_values(next_position, net_context.target_net)
            qv = torch.max(next_q_values).item()

        backpropagate(net_context, position, move_index, discount_factor * qv)

        next_position = position

    net_context.target_net.load_state_dict(net_context.policy_net.state_dict())


def backpropagate(net_context, position, move_index, target_value):
    net_context.optimizer.zero_grad()
    output = net_context.policy_net(convert_to_tensor(position))

    target = output.clone().detach()
    target[move_index] = target_value
    illegal_move_indexes = position.get_illegal_move_indexes()
    for mi in illegal_move_indexes:
        target[mi] = LOSS_VALUE

    loss = net_context.loss_function(output, target)
    loss.backward()
    net_context.optimizer.step()


def create_training_player(net_context, move_history, epsilon):
    def play(board):
        model = net_context.policy_net
        move_index = choose_move_index(board, model, epsilon)
        move_history.appendleft((board, move_index))
        updated_board = board.play_move(move_index)

        return updated_board

    return play


def choose_move_index(board, model, epsilon):
    if epsilon > 0:
        random_value_from_0_to_1 = np.random.uniform()
        if random_value_from_0_to_1 < epsilon:
            return randrange(9)

    with torch.no_grad():
        q_values = get_q_values(board, model)
        max_move_index = torch.argmax(q_values).item()

    return max_move_index


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
