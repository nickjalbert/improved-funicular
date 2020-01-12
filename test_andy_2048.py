import random
import numpy as np
from envs.andy_2048 import BoardEnv
from andy_adapter import Andy2048


def test_boardenv_random_direction():
    for _ in range(50):
        assert BoardEnv.random_direction() in [0, 1, 2, 3]


def test_boardenv_init():
    board_width = random.randint(4, 10)
    num_filled_init = random.randint(0, 4)
    b = BoardEnv(width=board_width, init_spots_filled=num_filled_init)
    num_non_zero_spots = (b.state != 0).sum().sum()
    assert num_non_zero_spots == num_filled_init, (
        "BoardEnv initializing wrong num spots %s" % num_non_zero_spots
    )


def test_boardenv_from_init_state():
    b = BoardEnv.from_init_state([[0, 0], [2, 0]])
    assert b.value == 0.0
    assert np.sum(b.state) == 2
    assert b.width == 2
    assert b.init_spots_filled == 1


def test_board_env_step_one():
    # make sure the behavior is correct when a row is full of same values.
    init_state = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert state[0, 3] == 2.0 and state[2, 3] == 2.0


def test_board_env_step_two():
    init_state = [
        [4.0, 2.0, 2.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert state[0, 1] == 4.0
    assert state[0, 2] == 4.0
    assert state[0, 3] == 4.0


# FIXME - is this different than test_board_env_step_one?
def test_board_env_step_three():
    init_state = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert state[0, 3] == 2.0 and state[2, 3] == 2.0, state


def test_boardenv_move_logic_four_in_a_row():
    # make sure the behavior is correct when a row is full of same values.
    init_state = [
        [2.0, 2.0, 2.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    assert np.array_equal(init_state, b.state)
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert reward == 8
    assert state[0, 2] == 4 and state[0, 3] == 4, b.state
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert reward >= 8
    assert state[0, 3] == 8, b.state


def test_boardenv_done_logic():
    init_state = [
        [16.0, 8.0, 16.0, 4.0],
        [4.0, 2.0, 4.0, 8.0],
        [32.0, 2.0, 32.0, 4.0],
        [4.0, 16.0, 4.0, 8.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done, _ = b.step(BoardEnv.RIGHT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done, _ = b.step(BoardEnv.LEFT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done, _ = b.step(BoardEnv.DOWN)
    assert done, state
    assert reward == 4


def test_boardenv_move_logic_three_in_a_row():
    # make sure the behavior is correct when 3 elts are same in a row
    init_state = [
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    assert np.array_equal(init_state, b.state)
    state, reward, done, _ = b.step(BoardEnv.DOWN)
    assert reward == 4
    assert state[3, 1] == 4 and state[2, 1] == 2, b.state


def test_boardenv_fill_on_move_logic():
    # make sure a new piece is added that is either a 2 or a 4
    init_state = [
        [2.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done, _ = b.step(BoardEnv.LEFT)
    num_non_zero_spots = (b.state != 0).sum().sum()
    assert num_non_zero_spots == 2, state


def test_set_board_makes_copy():
    init_state = [2, 2, 0, 0, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    game = Andy2048()
    game.set_board(init_state)
    assert game.board == init_state
    assert not (game.board is init_state)
