""" An implementation of 2048 that is mostly like a Gym Environment, except its board
    state and observations are in python tuples instead of numpy arrays."""
import random
from envs.base_2048 import Base2048
from etc.squash_lookup_table import squash_lookup
from gym.spaces import Discrete, Box
import numpy as np

# self.board is a 1D list that represents the 2D board follows:
#       [
#           board[0]   board[1]   board[2]   board[3]
#           board[4]   board[5]   board[6]   board[7]
#           board[8]   board[9]  board[10]  board[11]
#          board[12]  board[13]  board[14]  board[15]
#       ]
#
# TODO: allow for boards of different dimensions


class Nick2048(Base2048):
    info = "Nick's implementation of 2048"
    RIGHT = 0
    DOWN = 1
    LEFT = 2
    UP = 3
    action_space = Discrete(4)
    observation_space = Box(low=0, high=2 ** 30, shape=(16,), dtype=np.uint32)

    def __init__(self, config=None):
        self.reset()

    @property
    def done(self):
        if len(self.empty_indexes) > 0:
            return False
        # board is full, so just check neighbors to see if we can squish
        idx0 = self.board[0]
        idx1 = self.board[1]
        if idx0 == idx1:
            return False
        idx4 = self.board[4]
        if idx0 == idx4:
            return False
        idx2 = self.board[2]
        if idx1 == idx2:
            return False
        idx5 = self.board[5]
        if idx1 == idx5:
            return False
        idx3 = self.board[3]
        if idx2 == idx3:
            return False
        idx6 = self.board[6]
        if idx2 == idx6:
            return False
        idx7 = self.board[7]
        if idx3 == idx7:
            return False
        if idx4 == idx5:
            return False
        idx8 = self.board[8]
        if idx4 == idx8:
            return False
        if idx5 == idx6:
            return False
        idx9 = self.board[9]
        if idx5 == idx9:
            return False
        if idx6 == idx7:
            return False
        idx10 = self.board[10]
        if idx6 == idx10:
            return False
        idx11 = self.board[11]
        if idx7 == idx11:
            return False
        if idx8 == idx9:
            return False
        idx12 = self.board[12]
        if idx8 == idx12:
            return False
        idx13 = self.board[13]
        if idx9 == idx10:
            return False
        if idx9 == idx13:
            return False
        if idx10 == idx11:
            return False
        idx14 = self.board[14]
        if idx10 == idx14:
            return False
        idx15 = self.board[15]
        if idx11 == idx15:
            return False
        if idx12 == idx13:
            return False
        if idx13 == idx14:
            return False
        if idx14 == idx15:
            return False
        return True

    @classmethod
    def random_direction(cls):
        return random.choice([cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT])

    def get_state(self):
        return self.board, self.score, self.done

    def set_board(self, board):
        # Copy board as we set so we don't get surprising aliasing errors
        assert len(board) == 16
        self.empty_indexes = [i for (i, v) in enumerate(board) if v == 0]
        self.board = tuple(board)

    def step(self, action):
        """Returns a 3-tuple of (board, reward for action, boolean is_done)"""
        assert action in self.action_space
        do_action = {
            self.UP: self._do_up,
            self.RIGHT: self._do_right,
            self.DOWN: self._do_down,
            self.LEFT: self._do_left,
        }
        old_board = self.board
        old_score = self.score
        do_action[action]()
        if old_board != self.board:
            self.add_new_random_number()
        board, new_score, done = self.get_state()
        return board, new_score - old_score, done, {}

    def reset(self):
        self.set_board((0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        self.add_new_random_number()
        self.add_new_random_number()
        self.score = 0
        return self.board

    def add_new_random_number(self):
        return self._set_random_position(2 if random.random() <= 0.9 else 4)

    def _set_random_position(self, value):
        try:
            idx = random.choice(self.empty_indexes)
        except IndexError:
            return -1
        board = self.board[:idx] + (value,) + self.board[idx + 1:]
        self.set_board(board)
        return idx

    def _do_up(self):
        sq1, r1 = squash_lookup[(self.board[0], self.board[4], self.board[8], self.board[12])]
        sq2, r2 = squash_lookup[(self.board[1], self.board[5], self.board[9], self.board[13])]
        sq3, r3 = squash_lookup[(self.board[2], self.board[6], self.board[10], self.board[14])]
        sq4, r4 = squash_lookup[(self.board[3], self.board[7], self.board[11], self.board[15])]
        self.set_board((sq1[0], sq2[0], sq3[0], sq4[0],
                        sq1[1], sq2[1], sq3[1], sq4[1],
                        sq1[2], sq2[2], sq3[2], sq4[2],
                        sq1[3], sq2[3], sq3[3], sq4[3]))
        self.score += r1 + r2 + r3 + r4

    def _do_right(self):
        sq1, r1 = squash_lookup[(self.board[3], self.board[2], self.board[1], self.board[0])]
        sq2, r2 = squash_lookup[(self.board[7], self.board[6], self.board[5], self.board[4])]
        sq3, r3 = squash_lookup[(self.board[11], self.board[10], self.board[9], self.board[8])]
        sq4, r4 = squash_lookup[(self.board[15], self.board[14], self.board[13], self.board[12])]
        self.set_board((sq1[3], sq1[2], sq1[1], sq1[0],
                        sq2[3], sq2[2], sq2[1], sq2[0],
                        sq3[3], sq3[2], sq3[1], sq3[0],
                        sq4[3], sq4[2], sq4[1], sq4[0]))
        self.score += r1 + r2 + r3 + r4

    def _do_down(self):
        sq1, r1 = squash_lookup[(self.board[12], self.board[8], self.board[4], self.board[0])]
        sq2, r2 = squash_lookup[(self.board[13], self.board[9], self.board[5], self.board[1])]
        sq3, r3 = squash_lookup[(self.board[14], self.board[10], self.board[6], self.board[2])]
        sq4, r4 = squash_lookup[(self.board[15], self.board[11], self.board[7], self.board[3])]
        self.set_board((sq1[3], sq2[3], sq3[3], sq4[3],
                        sq1[2], sq2[2], sq3[2], sq4[2],
                        sq1[1], sq2[1], sq3[1], sq4[1],
                        sq1[0], sq2[0], sq3[0], sq4[0]))
        self.score += r1 + r2 + r3 + r4

    def _do_left(self):
        sq1, r1 = squash_lookup[(self.board[0], self.board[1], self.board[2], self.board[3])]
        sq2, r2 = squash_lookup[(self.board[4], self.board[5], self.board[6], self.board[7])]
        sq3, r3 = squash_lookup[(self.board[8], self.board[9], self.board[10], self.board[11])]
        sq4, r4 = squash_lookup[(self.board[12], self.board[13], self.board[14], self.board[15])]
        self.set_board((sq1[0], sq1[1], sq1[2], sq1[3],
                        sq2[0], sq2[1], sq2[2], sq2[3],
                        sq3[0], sq3[1], sq3[2], sq3[3],
                        sq4[0], sq4[1], sq4[2], sq4[3]))
        self.score += r1 + r2 + r3 + r4
