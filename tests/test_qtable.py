import pytest

import numpy as np

import random

from tictac.board_cache import BoardCache
from tictac.qtable import (get_q_values, get_noisy_q_values, get_action_index,
                           train_episode)


@pytest.fixture(autouse=True)
def seed_random_number_generators():
    random.seed(0)
    np.random.seed(0)


def test_get_q_values_initial():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()

    q_values = get_q_values(q_table, board)

    assert np.array_equal(q_values, np.zeros(4))


def test_get_noisy_q_values_initial_no_noise():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_values = get_q_values(q_table, board)

    noisy_q_values = get_noisy_q_values(q_values, 0)

    assert np.array_equal(noisy_q_values, np.zeros(4))


def test_get_noisy_q_values_initial_full_noise():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_values = get_q_values(q_table, board)

    noisy_q_values = get_noisy_q_values(q_values, 1)

    expected_q_values = np.array(
        [1.76405235, 0.40015721, 0.97873798, 2.2408932])

    actual_expected_pairs = zip(noisy_q_values, expected_q_values)
    for actual, expected in actual_expected_pairs:
        assert round(actual, 8) == expected


def test_get_noisy_q_values_initial_1_10th_noise():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_values = get_q_values(q_table, board)

    noisy_q_values = get_noisy_q_values(q_values, 0.1)

    expected_q_values = np.array(
        [1.76405235/10, 0.40015721/10, 0.97873798/10, 2.2408932/10])

    actual_expected_pairs = zip(noisy_q_values, expected_q_values)
    for actual, expected in actual_expected_pairs:
        assert round(actual, 7) == round(expected, 7)


def test_get_noisy_q_values_use_prior_values_1_10th_noise():
    board = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0,  0]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([1, 2, 3, 4]))
    q_values = get_q_values(q_table, board)

    noisy_q_values = get_noisy_q_values(q_values, 0.1)

    expected_q_values = np.array(
        [1+1.76405235/10, 2+0.40015721/10, 3+0.97873798/10, 4+2.2408932/10])

    actual_expected_pairs = zip(noisy_q_values, expected_q_values)
    for actual, expected in actual_expected_pairs:
        assert round(actual, 7) == round(expected, 7)


def test_get_action_index_choose_1st_move():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([-0.8, 0.5]))

    action_index = get_action_index(q_table, board, 0)

    assert action_index == 0


def test_get_action_index_choose_2nd_move():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([0.5, -0.8]))

    action_index = get_action_index(q_table, board, 0)

    assert action_index == 1


def test_train_episode_end_of_game():
    board = np.array([[ 1,  0,  0],
                      [ 1, -1,  1],
                      [-1,  1, -1]]).reshape(1, 9)[0]

    q_table = BoardCache()
    q_table.set_for_position(board, np.array([0.5, -0.8]))

    train_episode(q_table, board, 0.9, 0.9, 0.0)

    q_values, found = q_table.get_for_position(board)

    assert found
    assert np.array_equal(q_values, np.array([0.5, -1.]))


def test_train_episode_last_two_moves():
    board1 = np.array([[ 0,  0,  1],
                       [ 1, -1,  1],
                       [-1,  0, -1]]).reshape(1, 9)[0]
    q_table = BoardCache()
    q_table.set_for_position(board1, np.array([0.5, 0, 0]))

    board2 = np.array([[ 1,  0,  1],
                       [ 1, -1,  1],
                       [-1,  0, -1]]).reshape(1, 9)[0]
    q_table.set_for_position(board2, np.array([0, -0.5]))

    train_episode(q_table, board1, 1.0, 1.0, 0.0)

    board1_q_values, found = q_table.get_for_position(board1)
    assert found
    assert np.array_equal(board1_q_values, np.array([0., 0., 0.]))

    board2_q_values, found = q_table.get_for_position(board2)
    assert found
    assert np.array_equal(board2_q_values, np.array([0., -1.]))
