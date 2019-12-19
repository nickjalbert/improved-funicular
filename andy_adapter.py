from andy_2048 import BoardEnv, Direction


class Andy2048:
    UP = Direction.U
    RIGHT = Direction.R
    DOWN = Direction.D
    LEFT = Direction.L

    def __init__(self):
        self.andy = BoardEnv()

    @property
    def board(self):
        board = []
        for row in self.andy.state:
            for el in row:
                board.append(el)
        return board

    @property
    def score(self):
        return self.andy.value

    def step(self, direction):
        return self.andy.step(direction)
