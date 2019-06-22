from tictac.common import get_board_2d, get_symmetrical_board_orientations


class BoardCache():
    def __init__(self):
        self.cache = {}

    def set_for_position(self, board, value):
        board_2d = get_board_2d(board)
        self.cache[board_2d.tobytes()] = value

    def get_for_position(self, board):
        board_2d = get_board_2d(board)
        board_orientations = get_symmetrical_board_orientations(board_2d)

        for b in board_orientations:
            result = self.cache.get(b.tobytes())
            if result is not None:
                return result, True

        return None, False

    def reset(self):
        self.cache = {}
