import numpy as np

from tictac.common import (play_move, get_valid_move_indexes)


def play_random_move(board):
    move = get_random_valid_move(board)
    return play_move(board, move)


def get_random_valid_move(board):
    valid_move_indexes = get_valid_move_indexes(board)
    random_move = np.random.choice(valid_move_indexes)
    return random_move
