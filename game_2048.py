import random
import readchar


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
        return self.board

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
            f"{p[12]} {p[13]} {p[14]} {p[15]}\n\n"
        )

    def run_manual_loop(self):
        game.render_board()

        moves = {
            "w": Game2048.UP,
            "A": Game2048.UP,
            "d": Game2048.RIGHT,
            "C": Game2048.RIGHT,
            "s": Game2048.DOWN,
            "B": Game2048.DOWN,
            "a": Game2048.LEFT,
            "D": Game2048.LEFT,
        }
        should_print = True
        while True:
            if should_print:
                print("Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?")
                should_print = False
            move = readchar.readchar()
            if move in ["\x03", "\x04", "\x1a"]:
                print("Exiting...")
                break
            if move not in moves:
                continue
            should_print = True
            game.step(moves[move])
            game.render_board()

    def _set_random_position(self, value):
        idxs = [i for i, val in enumerate(self.board) if val is None]
        if not idxs:
            return None
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
        new_col = [v for v in col if v is not None]
        i = 0
        while i < len(new_col) - 1:
            if new_col[i] == new_col[i + 1]:
                new_col[i] *= 2
                new_col.pop(i + 1)
            i += 1

        while len(new_col) < len(col):
            new_col.append(None)
        return new_col


if __name__ == "__main__":
    game = Game2048()
    game.run_manual_loop()
