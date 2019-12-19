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
        self.andy.state = [
            [board[0], board[1], board[2], board[3]],
            [board[4], board[5], board[6], board[7]],
            [board[8], board[9], board[10], board[11]],
            [board[12], board[13], board[14], board[15]],
        ]
