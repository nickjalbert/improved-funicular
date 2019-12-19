import sys
import statistics
import random
from nick_2048 import Nick2048

TRIALS = 1000


def do_trials(strategy, check_done_fn=None):
    scores = []
    max_tiles = []
    for i in range(TRIALS):
        game = Nick2048()
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


def try_only_go_right():
    def right_fn(board):
        return Nick2048.RIGHT

    def right_done(prev, curr, score, done):
        return done or prev == curr

    right_fn.info = "Strategy only moves right"
    do_trials(right_fn, right_done)


def try_random():
    def random_fn(board):
        choices = [Nick2048.UP, Nick2048.RIGHT, Nick2048.DOWN, Nick2048.LEFT]
        return random.choice(choices)

    random_fn.info = "Random strategy"
    do_trials(random_fn)


def try_down_left():
    def down_left_fn(board):
        game = Nick2048()
        game.board = board
        down_left_actions = [Nick2048.DOWN, Nick2048.LEFT]
        right_up_actions = [Nick2048.RIGHT, Nick2048.UP]
        random.shuffle(down_left_actions)
        random.shuffle(right_up_actions)
        for action in down_left_actions + right_up_actions:
            game.board = board[:]
            assert game.board == board
            game.step(action)
            if game.board != board:
                return action
        assert False, "Some action should do something"

    down_left_fn.info = "Down Left strategy"

    do_trials(down_left_fn)


def do_stats():
    print(f"\nRunning {TRIALS} trials to test each strategy\n")
    try_only_go_right()
    try_random()
    try_down_left()


if __name__ == "__main__":
    do_stats()
