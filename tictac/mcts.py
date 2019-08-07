import math

from tictac.board import play_game
from tictac.board import (Board, BoardCache, CELL_X, CELL_O, RESULT_X_WINS,
                          RESULT_O_WINS, is_draw)

nodecache = BoardCache()


class Node:
    def __init__(self):
        self.parents = BoardCache()
        self.visits = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def add_parent_node(self, node_cache, parent_board):
        result, found = self.parents.get_for_position(parent_board)
        if found is False:
            parent_node = find_or_create_node(node_cache, parent_board)
            self.parents.set_for_position(parent_board, parent_node)

    def get_total_visits_for_parent_nodes(self):
        return sum([parent_node.visits for parent_node
                    in self.parents.cache.values()])

    def value(self):
        if self.visits == 0:
            return 0

        success_percentage = (self.wins + self.draws) / self.visits
        return success_percentage


def play_game_and_reset_playouts(x_strategy, o_strategy, node_cache=nodecache):
    node_cache.reset()
    board = play_game(x_strategy, o_strategy)
    node_cache.reset()
    return board

def play_mcts_move_with_live_playouts(board, node_cache=nodecache, num_playouts=200):
    perform_training_playouts(node_cache, board, num_playouts,
                              display_progress=False)
    return play_mcts_move(board, node_cache)

def play_mcts_move(board, node_cache=nodecache):
    move_index_node_pairs = get_move_index_node_pairs(board, node_cache)
    move_index_to_play = max(move_index_node_pairs,
                             key=lambda pair: pair[1].value())[0]
    return board.play_move(move_index_to_play)


def get_move_index_node_pairs(board, node_cache):
    boards = [board.play_move(mi) for mi in board.get_valid_move_indexes()]
    nodes = [find_or_create_node(node_cache, b) for b in boards]

    return zip(board.get_valid_move_indexes(), nodes)


def perform_training_playouts(node_cache=nodecache, board=Board(),
                              num_playouts=4000, display_progress=True):
    for game in range(num_playouts):
        perform_game_playout(node_cache, board)
        if display_progress is True and (game+1) % (num_playouts / 10) == 0:
            print(f"{game+1}/{num_playouts} playouts...")


def perform_game_playout(node_cache, board):
    game_history = [board]

    while not board.is_gameover():
        move_index = choose_move(node_cache, board)
        board = board.play_move(move_index)
        game_history.append(board)

    backpropagate(node_cache, board, game_history)


def choose_move(node_cache, parent_board):
    move_value_pairs = calculate_values(node_cache, parent_board)
    return max(move_value_pairs, key=lambda pair: pair[1])[0]


def calculate_values(node_cache, parent_board):
    child_boards = [parent_board.play_move(mi) for mi
                    in parent_board.get_valid_move_indexes()]
    values = [calculate_value(node_cache, parent_board, cb) for cb
              in child_boards]
    return zip(parent_board.get_valid_move_indexes(), values)


def calculate_value(node_cache, parent_board, board):
    node = find_or_create_node(node_cache, board)
    node.add_parent_node(node_cache, parent_board)
    if node.visits == 0:
        return math.inf

    parent_node_visits = node.get_total_visits_for_parent_nodes()

    assert node.visits <= parent_node_visits, \
        "child node visits should be a subset of visits to the parent node "

    exploration_term = (math.sqrt(2.0)
                        * math.sqrt(math.log(parent_node_visits) / node.visits))

    value = node.value() + exploration_term

    return value


def backpropagate(node_cache, final_board_position, game_history):
    for board in game_history:
        node = find_node(node_cache, board)
        node.visits += 1
        if is_win(board.get_turn(), final_board_position):
            node.wins += 1
        elif is_loss(board.get_turn(), final_board_position):
            node.losses += 1
        elif is_draw(final_board_position):
            node.draws += 1
        else:
            raise ValueError("Illegal game state")


def find_node(node_cache, board):
    result, found = node_cache.get_for_position(board)
    assert found is True, "node must exist"
    node, _ = result
    return node


def find_or_create_node(node_cache, board):
    result, found = node_cache.get_for_position(board)
    if found is False:
        node = Node()
        node_cache.set_for_position(board, node)
        return node

    node, _ = result
    return node


def is_win(player, board):
    result = board.get_game_result()
    return ((player == CELL_X and result == RESULT_O_WINS)
            or (player == CELL_O and result == RESULT_X_WINS))


def is_loss(player, board):
    result = board.get_game_result()
    return ((player == CELL_X and result == RESULT_X_WINS)
            or (player == CELL_O and result == RESULT_O_WINS))
