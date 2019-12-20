# copied from https://colab.research.google.com/drive/1rp7kJrY-vkrCRziMJUatHL75dYVmlYv7
from enum import Enum
import random
import pandas as pd
import numpy as np


class BoardEnv:
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3

    def __init__(self, width=4, init_spots_filled=2):
        self.action_space = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        assert 0 <= init_spots_filled <= width ** 2
        self.value = 0
        self.width = width
        self.state = np.array([[0.0] * width] * width)
        self.init_spots_filled = init_spots_filled
        indices = [((x, y)) for x in range(width) for y in range(width)]
        for i in range(init_spots_filled):
            rand_index = indices.pop(random.randint(0, len(indices) - 1))
            val = 2.0 if random.random() > 0.1 else 4.0
            self.state[rand_index[0], rand_index[1]] = val

    @classmethod
    def from_init_state(cls, state_array):
        state = np.array(state_array)
        assert len(state.shape) == 2 and state.shape[0] == state.shape[1]
        width = state.shape[1]
        init_spots_filled = np.count_nonzero(state)
        b = cls(width, init_spots_filled)
        b.state = state
        return b

    @classmethod
    def random_direction(cls):
        return np.random.randint(4)

    def __str__(self):
        return pd.DataFrame(self.state).to_string()

    @property
    def done(self):
        for d in [self.RIGHT, self.DOWN, self.LEFT, self.UP]:
            new_board, _ = self.try_move(self.state.copy(), d)
            if not np.array_equal(new_board, self.state):
                return False
        return True

    def reset(self):
        self.__init__(self.width, self.init_spots_filled)
        return self.state.copy()

    def try_move(self, state, direction):
        # rotate the board so we only have to implement the shifting logic for one direction.
        # we will rotate it back later after we shift all the pieces.
        state = np.rot90(m=state, k=direction)
        reward = 0.0
        stop_walls = [self.width] * self.width
        # handle one col at a time time, R, to L: (3, 2, 1, 0)
        for curr_col in range(self.width - 1, -1, -1):
            # handle one row at at time
            for curr_row in range(self.width):
                curr_val = state[curr_row, curr_col]
                check_col = curr_col + 1
                new_col = None
                new_val = 0.0
                # slide the piece in each row to the right (iterate check_spot to the right)
                # stop when we get to the edge of the board or else a non-zero position.
                while (
                        check_col < self.width
                        and check_col < stop_walls[curr_row]
                        and curr_val != 0
                ):
                    check_val = state[curr_row, check_col]
                    # we will want to slide the cur position over top of this zero
                    if check_val == 0.0:
                        new_col = check_col
                        new_val = curr_val
                    else:  # you can't slide any further, you hit a non-zero.
                        # this is same num as sliding tile, so merge
                        if check_val == curr_val:
                            new_col = check_col
                            new_val = curr_val * 2.0
                            reward += new_val
                            # don't let anything merge into this again.
                            stop_walls[curr_row] = check_col
                        break  # hit a non-zero, stop checking to the right for this (curr_col, curr_row) piece
                    check_col += 1
                if new_col:
                    # merge this piece into new position
                    state[curr_row, new_col] = new_val
                    # make sure its former position is empty
                    state[curr_row, curr_col] = 0
        # rotate our state back
        return np.rot90(m=state, k=(-1 * direction)), reward

    # takes a direction to move the board
    # returns tuple (next_state, reward, done)
    def step(self, direction):
        new_board, reward = self.try_move(self.state.copy(), direction)
        self.value += reward
        # if the move they attempted resulted in at least one tile moving,
        # add a new tile in random spot
        if not np.array_equal(new_board, self.state):
            indices = [(x, y) for x in range(self.width) for y in range(self.width)]
            while indices:
                rand_index = indices.pop(random.randint(0, len(indices) - 1))
                if new_board[rand_index[0], rand_index[1]] == 0.0:
                    val = 2.0 if random.random() > 0.1 else 4.0
                    new_board[rand_index[0], rand_index[1]] = val
                    break
        self.state = new_board
        return new_board, reward, self.done

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
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert state[0, 3] == 2.0 and state[2, 3] == 2.0

def test_board_env_step_two():
    init_state = [
        [4.0, 2.0, 2.0, 4.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert state[0, 1] == 4.0
    assert state[0, 2] == 4.0
    assert state[0, 3] == 4.0

def test_board_env_step_three():
    init_state = [
        [2.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done = b.step(BoardEnv.RIGHT)
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
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert reward == 8
    assert state[0, 2] == 4 and state[0, 3] == 4, b.state
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert reward >= 8
    assert state[0, 3] == 8, b.state

def test_boardenv_done_logic():
    init_state = [
        [16.0,  8.0, 16.0, 4.0],
        [ 4.0,  2.0,  4.0, 8.0],
        [32.0,  2.0, 32.0, 4.0],
        [ 4.0, 16.0,  4.0, 8.0],
    ]
    b = BoardEnv().from_init_state(init_state)
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done = b.step(BoardEnv.RIGHT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done = b.step(BoardEnv.LEFT)
    assert not done and np.array_equal(state, np.array(init_state))
    assert reward == 0
    state, reward, done = b.step(BoardEnv.DOWN)
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
    state, reward, done = b.step(BoardEnv.DOWN)
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
    state, reward, done = b.step(BoardEnv.LEFT)
    num_non_zero_spots = (b.state != 0).sum().sum()
    assert num_non_zero_spots == 2, state


def test_boardenv_init():
    b = BoardEnv.from_init_state([[2, 0], [2, 0]])
    _, reward, _ = b.step(BoardEnv.DOWN)
    assert reward == 4
