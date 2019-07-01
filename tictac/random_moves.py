def play_random_move(board):
    move = board.get_random_valid_move_index()
    return board.play_move(move)
