import sys
import statistics
import random
from nick_2048 import Nick2048
from andy_adapter import Andy2048

TRIALS = 10000


def do_trials(cls, strategy, check_done_fn=None):
    scores = []
    max_tiles = []
    for i in range(TRIALS):
        if i % 1000 == 0:
            print("iter %s" % i)
        game = cls()
        curr_board, score, done = game.get_state()
        while not done:
            assert curr_board == game.board
            move = strategy(curr_board)
            prev_board = curr_board[:]
            curr_board, score, done = game.step(move)
            if check_done_fn is not None:
                done = check_done_fn(prev_board, curr_board, score, done)
        scores.append(score)
        max_tiles.append(max(game.board))
    print(
        f"{strategy.info}:\n"
        f"\tMax Tile: {max(max_tiles)}\n"
        f"\tMax Score: {max(scores)}\n"
        f"\tMean Score: {statistics.mean(scores)}\n"
        f"\tMedian Score: {statistics.median(scores)}\n"
        f"\tStandard Dev: {statistics.stdev(scores)}\n"
        f"\tMin Score: {min(scores)}\n"
    )

def try_only_go_right(cls):
    def right_fn(board):
        return cls.RIGHT

    def right_done(prev, curr, score, done):
        return done or prev == curr

    right_fn.info = "Strategy only moves right"
    do_trials(cls, right_fn, right_done)


def try_random(cls):
    def random_fn(board):
        choices = [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]
        return random.choice(choices)

    random_fn.info = "Random strategy"
    do_trials(cls, random_fn)


def try_down_left(cls):
    def down_left_fn(board):
        game = cls()
        game.set_board(board)
        down_left_actions = [cls.DOWN, cls.LEFT]
        right_up_actions = [cls.RIGHT, cls.UP]
        random.shuffle(down_left_actions)
        random.shuffle(right_up_actions)
        for action in down_left_actions + right_up_actions:
            game.set_board(board[:])
            assert game.board == board
            game.step(action)
            if game.board != board:
                return action
        assert False, "Some action should do something"

    down_left_fn.info = "Down Left strategy"
    do_trials(cls, down_left_fn)


def do_stats():
    #print(f"\nRunning {TRIALS} trials with Nick impl to test each strategy\n")
    #try_only_go_right(Nick2048)
    #try_random(Nick2048)
    try_down_left(Nick2048)
    #print(f"\nRunning {TRIALS} trials with Andy impl to test each strategy\n")
    #try_only_go_right(Andy2048)
    #try_random(Andy2048)
    try_down_left(Andy2048)



if __name__ == "__main__":
    do_stats()
