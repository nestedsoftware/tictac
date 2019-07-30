import math

from tictac.board import (Board, BoardCache, CELL_X, CELL_O, RESULT_X_WINS,
                          RESULT_O_WINS, is_draw)

nodecache = BoardCache()


class Node:
    def __init__(self):
        self.visits = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def value(self):
        if self.visits == 0:
            return 0

        percentage_wins = float(self.wins + self.draws)/self.visits
        return percentage_wins


def play_mcts_move(board, node_cache=nodecache, num_playouts=10):
    perform_training_playouts(node_cache, board, num_playouts)
    move_index_node_pairs = get_move_index_node_pairs(board, node_cache)

    move_index_to_play = max(move_index_node_pairs,
                             key=lambda pair: pair[1].value())[0]
    return board.play_move(move_index_to_play)


def get_move_index_node_pairs(board, node_cache):
    nodes = []
    valid_move_indexes = board.get_valid_move_indexes()
    for move_index in valid_move_indexes:
        b = board.play_move(move_index)
        n = find_or_create_node(node_cache, b)
        nodes.append((move_index, n))
    return nodes


def perform_training_playouts(node_cache=nodecache, board=Board(),
                              num_playouts=30000):
    for game in range(num_playouts):
        perform_game_playout(node_cache, board)

        # if (game+1) % (num_playouts / 10) == 0:
        #     print(f"{game+1} playouts...")


def perform_game_playout(node_cache, board):
    game_history = []

    while not board.is_gameover():
        parent_node = find_or_create_node(node_cache, board)
        parent_node.visits += 1
        move_index_node_pairs = find_or_create_child_nodes(node_cache, board)
        move_index = choose_node(parent_node, move_index_node_pairs)
        board = board.play_move(move_index)
        game_history.append(board)

    final_node = find_or_create_node(node_cache, board)
    final_node.visits += 1
    backpropagate(node_cache, board, game_history)


def choose_node(parent_node, move_index_node_pairs):
    move_value_pairs = calculate_values(parent_node, move_index_node_pairs)

    return max(move_value_pairs, key=lambda pair: pair[1])[0]


def calculate_values(parent_node, move_index_node_pairs):
    values = []
    for (move_index, node) in move_index_node_pairs:
        value = calculate_value(parent_node, node)
        values.append((move_index, value))
    return values


def calculate_value(parent_node, node):
    if node.visits == 0:
        return math.inf

    value = (node.value()
             + (math.sqrt(2.0)
             * math.sqrt(math.log(parent_node.visits) / node.visits)))

    return value


def backpropagate(node_cache, final_board_position, game_history):
    for b in game_history:
        result, found = node_cache.get_for_position(b)
        assert found, "must be cached"
        node, _ = result
        if is_win(b.get_turn(), final_board_position):
            node.wins += 1
        elif is_loss(b.get_turn(), final_board_position):
            node.losses += 1
        elif is_draw(final_board_position):
            node.draws += 1


def find_or_create_node(node_cache, board):
    result, found = node_cache.get_for_position(board)
    if found is False:
        node = Node()
        node_cache.set_for_position(board, node)
        return node

    node, _ = result
    return node


def find_or_create_child_nodes(node_cache, board):
    valid_move_indexes = board.get_valid_move_indexes()
    child_nodes = []
    for move_index in valid_move_indexes:
        b = board.play_move(move_index)
        node = find_or_create_node(node_cache, b)
        child_nodes.append((move_index, node))

    return child_nodes


def is_win(player, board):
    result = board.get_game_result()
    return ((player == CELL_X and result == RESULT_O_WINS)
            or (player == CELL_O and result == RESULT_X_WINS))


def is_loss(player, board):
    result = board.get_game_result()
    return ((player == CELL_X and result == RESULT_X_WINS)
            or (player == CELL_O and result == RESULT_O_WINS))
