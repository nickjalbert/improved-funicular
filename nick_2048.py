import random


class Nick2048:
    UP = "U"
    RIGHT = "R"
    DOWN = "D"
    LEFT = "L"

    def __init__(self):
        self.action_space = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        self.reset()

    @property
    def done(self):
        self.board = self.board[:]
        original_board = self.board[:]
        original_score = self.score
        is_done = True
        actions = [self._do_up, self._do_left, self._do_right, self._do_down]
        for action in actions:
            action()
            if original_board != self.board:
                is_done = False
                break
        self.board = original_board
        self.score = original_score
        return is_done

    def get_state(self):
        return self.board, self.score, self.done

    def set_board(self, board):
        self.board = board[:]

    def step(self, action):
        assert action in self.action_space
        do_action = {
            self.UP: self._do_up,
            self.RIGHT: self._do_right,
            self.DOWN: self._do_down,
            self.LEFT: self._do_left,
        }
        old_board = self.board[:]
        self.board = self.board[:]
        do_action[action]()
        if old_board != self.board:
            self.add_new_random_number()
        return self.get_state()

    def reset(self):
        self.board = [0] * 16
        self.add_new_random_number()
        self.add_new_random_number()
        self.score = 0

    def add_new_random_number(self):
        self._set_random_position(2 if random.random() <= 0.85 else 4)

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
