import random
from base_2048 import Base2048

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

    def __init__(self, init_board=None, init_score=0):
        self.action_space = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        self.reset(init_board=init_board, init_score=init_score)

    @property
    def done(self):
        if len([v for v in self.board if v == 0]) > 0:
            return False
        # board is full, so just check neighbors to see if we can squish
        check_indices = {
            0: [1, 4],
            1: [0, 2, 5],
            2: [1, 3, 6],
            3: [2, 7],
            4: [0, 5, 8],
            5: [1, 4, 6, 9],
            6: [2, 5, 7, 10],
            7: [3, 6, 11],
            8: [4, 9, 12],
            9: [5, 8, 10, 13],
            10: [6, 9, 11, 14],
            11: [7, 10, 15],
            12: [8, 13],
            13: [9, 12, 14],
            14: [10, 13, 15],
            15: [11, 14],
        }
        for (idx, check_idxs) in check_indices.items():
            el = self.board[idx]
            for check_idx in check_idxs:
                if el == self.board[check_idx]:
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
        self.board = board[:]

    def step(self, action):
        """Returns a 3-tuple of (board, reward for action, boolean is_done)"""
        assert action in self.action_space
        do_action = {
            self.UP: self._do_up,
            self.RIGHT: self._do_right,
            self.DOWN: self._do_down,
            self.LEFT: self._do_left,
        }
        old_board = self.board[:]
        old_score = self.score
        do_action[action]()
        if old_board != self.board:
            self.add_new_random_number()
        board, new_score, done = self.get_state()
        return board, new_score - old_score, done

    def reset(self, init_board=None, init_score=0):
        if init_board:
            self.board = init_board.copy()
        else:
            self.board = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            self.add_new_random_number()
            self.add_new_random_number()
        self.score = init_score

    def add_new_random_number(self):
        self._set_random_position(2 if random.random() <= 0.9 else 4)

    def _set_random_position(self, value):
        idxs = [i for i, val in enumerate(self.board) if val == 0]
        if not idxs:
            return 0
        idx = random.choice(idxs)
        self.board[idx] = value
        return idx

    def _do_up(self):
        self._smush_by_index([0, 4, 8, 12])
        self._smush_by_index([1, 5, 9, 13])
        self._smush_by_index([2, 6, 10, 14])
        self._smush_by_index([3, 7, 11, 15])

    def _do_right(self):
        self._smush_by_index([3, 2, 1, 0])
        self._smush_by_index([7, 6, 5, 4])
        self._smush_by_index([11, 10, 9, 8])
        self._smush_by_index([15, 14, 13, 12])

    def _do_down(self):
        self._smush_by_index([12, 8, 4, 0])
        self._smush_by_index([13, 9, 5, 1])
        self._smush_by_index([14, 10, 6, 2])
        self._smush_by_index([15, 11, 7, 3])

    def _do_left(self):
        self._smush_by_index([0, 1, 2, 3])
        self._smush_by_index([4, 5, 6, 7])
        self._smush_by_index([8, 9, 10, 11])
        self._smush_by_index([12, 13, 14, 15])

    def _smush_by_index(self, idxs):
        smushed = self._smush_left([self.board[i] for i in idxs])
        for i in idxs:
            self.board[i] = smushed.pop(0)

    def _smush_left(self, col):
        new_col = [v for v in col if v != 0]
        i = 0
        while i < len(new_col) - 1:
            if new_col[i] == new_col[i + 1]:
                val = new_col[i] * 2
                self.score += val
                new_col[i] = val
                new_col.pop(i + 1)
            i += 1

        while len(new_col) < len(col):
            new_col.append(0)
        return new_col
