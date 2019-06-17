import numpy as np
import itertools

BOARD_SIZE = 3
BOARD_DIMENSIONS = (BOARD_SIZE, BOARD_SIZE)

CELL_X = 1
CELL_O = -1
CELL_EMPTY = 0

RESULT_X_WINS = 1
RESULT_O_WINS = -1
RESULT_DRAW = 0
RESULT_NOT_OVER = 2


def play_games(total_games, x_strategy, o_strategy):
    results = {
        RESULT_X_WINS: 0,
        RESULT_O_WINS: 0,
        RESULT_DRAW: 0
    }

    board = np.array([CELL_EMPTY] * BOARD_SIZE**2)

    for g in range(total_games):
        result = play_game(board, x_strategy, o_strategy)
        results[result] += 1

    x_wins_percent = results[RESULT_X_WINS] / total_games * 100
    o_wins_percent = results[RESULT_O_WINS] / total_games * 100
    draw_percent = results[RESULT_DRAW] / total_games * 100

    print(f"x wins: {x_wins_percent:.2f}%")
    print(f"o wins: {o_wins_percent:.2f}%")
    print(f"draw  : {draw_percent:.2f}%")


def play_game(board, x_strategy, o_strategy):
    player_strategies = itertools.cycle([x_strategy, o_strategy])

    while not is_gameover(board):
        play = next(player_strategies)
        board = play(board)

    return get_game_result(board)


def get_game_result(board):
    board_2d = get_board_2d(board)

    rows_cols_and_diagonals = get_rows_cols_and_diagonals(board_2d)

    sums = list(map(sum, rows_cols_and_diagonals))
    max_value = max(sums)
    min_value = min(sums)

    if max_value == BOARD_SIZE:
        return RESULT_X_WINS

    if min_value == -BOARD_SIZE:
        return RESULT_O_WINS

    if CELL_EMPTY not in board_2d:
        return RESULT_DRAW

    return RESULT_NOT_OVER


def is_gameover(board):
    return get_game_result(board) != RESULT_NOT_OVER


def play_move(board, move):
    turn = get_turn(board)
    board_copy = np.copy(board)
    board_copy[move] = turn
    return board_copy


def get_turn(board):
    non_zero = np.count_nonzero(board)
    return CELL_X if non_zero % 2 == 0 else CELL_O


def get_valid_move_indexes(board):
    return [i for i in range(board.size) if board[i] == CELL_EMPTY]


def get_rows_cols_and_diagonals(board_2d):
    rows_and_diagonal = get_rows_and_diagonal(board_2d)
    cols_and_antidiagonal = get_rows_and_diagonal(np.rot90(board_2d))
    return rows_and_diagonal + cols_and_antidiagonal


def get_rows_and_diagonal(board_2d):
    num_rows = board_2d.shape[0]
    return ([row for row in board_2d[range(num_rows), :]]
            + [board_2d.diagonal()])


def print_board(board):
    print(get_board_as_string(board))


def get_board_as_string(board):
    board2d = board.reshape(BOARD_DIMENSIONS)
    rows, cols = board2d.shape
    board_as_string = ("-------\n")
    for r in range(rows):
        for c in range(cols):
            move = get_symbol(board2d[r, c])
            if c == 0:
                board_as_string += f"|{move}|"
            elif c == 1:
                board_as_string += f"{move}|"
            else:
                board_as_string += f"{move}|\n"
    board_as_string += ("-------\n")

    return board_as_string


def get_symbol(cell):
    if cell == CELL_X:
        return 'X'
    if cell == CELL_O:
        return 'O'
    return '-'


def get_board_2d(board):
    return board.reshape(BOARD_DIMENSIONS)


def not_empty(items):
    return items is not None and len(items) > 0
