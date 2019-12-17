import random


class Game2048:
    UP = "U"
    RIGHT = "R"
    DOWN = "D"
    LEFT = "L"

    def __init__(self):
        self.action_space = [self.UP, self.RIGHT, self.DOWN, self.LEFT]
        self.reset()

    def step(self, action):
        assert action in self.action_space
        do_action = {
            self.UP: self._do_up,
            self.RIGHT: self._do_right,
            self.DOWN: self._do_down,
            self.LEFT: self._do_left,
            }
        do_action[action]()
        self._set_random_position(2)

    def reset(self):
        self.board = [None] * 16
        self._set_random_position(2)
        self._set_random_position(2)

    def render_board(self):
        p = ["·" if x is None else x for x in self.board]
        print(
            f"{p[0]} {p[1]} {p[2]} {p[3]}\n"
            f"{p[4]} {p[5]} {p[6]} {p[7]}\n"
            f"{p[8]} {p[9]} {p[10]} {p[11]}\n"
            f"{p[12]} {p[13]} {p[14]} {p[15]}\n"
        )

    def _set_random_position(self, value):
        idxs = [i for i, val in enumerate(self.board) if val is None]
        if not idxs:
            return None
        idx = random.choice(idxs)
        self.board[idx] = value
        return idx

    def _do_up(self):
        self._smush_by_index([0,4,8,12])
        self._smush_by_index([1,5,9,13])
        self._smush_by_index([2,6,10,14])
        self._smush_by_index([3,7,11,15])

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
        self._smush_by_index([12, 13, 14, 5])

    def _smush_by_index(self, idxs):
        smushed = self._smush_left([self.board[i] for i in idxs])
        for i in idxs:
            self.board[i] = smushed.pop(0)

    def _smush_left(self, col):
        print(col)
        new_col = []
        curr_idx = 0
        while curr_idx < len(col):
            curr = col[curr_idx]
            after_idx = curr_idx + 1
            # at end of column
            if after_idx >= len(col):
                new_col.append(curr)
                break

            # None cannot be smushed
            if curr is None:
                curr_idx += 1
                continue

            after = col[after_idx]
            if curr == after:
                # Smush current and after
                new_col.append(curr + after)
                curr_idx = after_idx + 1
            else:
                # Keep current
                new_col.append(curr)
                curr_idx += 1

        while len(new_col) < 4:
            new_col.append(None)
        return new_col


if __name__ == "__main__":
    game = Game2048()
    game.render_board()
    game.step(Game2048.UP)
    game.render_board()
