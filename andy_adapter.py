from andy_2048 import BoardEnv
import numpy as np


class Andy2048:
    UP = BoardEnv.UP
    RIGHT = BoardEnv.RIGHT
    DOWN = BoardEnv.DOWN
    LEFT = BoardEnv.LEFT

    @classmethod
    def from_init_state(cls, init_state):
        andy_wrapper = cls()
        if isinstance(init_state, list) and len(init_state) == 16:
            init_state =np.array(init_state).reshape((4,4))
        andy_wrapper.andy = BoardEnv.from_init_state(init_state)
        return andy_wrapper

    def __init__(self):
        self.andy = BoardEnv()

    @property
    def board(self):
        board = []
        for row in self.andy.state:
            for el in row:
                board.append(int(el))
        return board

    @property
    def score(self):
        return self.andy.value

    def step(self, direction):
        self.andy.step(direction)
        return self.board, self.score, self.andy.done

    def get_state(self):
        return self.board, self.score, self.andy.done

    def set_board(self, board):
        self.andy.state = np.array(board).reshape(4,4)
