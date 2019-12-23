import time
import random
import statistics
from nick_2048 import Nick2048
from andy_adapter import Andy2048

TRIALS = 1000


def do_trials(cls, strategy, check_done_fn=None):
    start_time = time.time()
    scores = []
    max_tiles = []
    for i in range(TRIALS):
        game = cls()
        curr_board, score, done = game.get_state()
        while not done:
            assert curr_board == game.board
            move = strategy(curr_board)
            prev_board = curr_board[:]
            curr_board, reward, done = game.step(move)
            if check_done_fn is not None:
                done = check_done_fn(prev_board, curr_board, reward, done)
        _, score, _ = game.get_state()
        scores.append(score)
        max_tiles.append(max(game.board))
    elapsed = time.time() - start_time
    elapsed_per_trial = elapsed / TRIALS
    elapsed = round(elapsed, 2)
    elapsed_per_trial = round(elapsed_per_trial, 5)
    print(
        f"{strategy.info} "
        f"({elapsed} sec total, {elapsed_per_trial} sec per trial):\n"
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

    def right_done(prev, curr, reward, done):
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
        test_game = cls()
        test_game.set_board(board)
        valid_actions = [a for (a, r) in test_game.get_valid_actions()]
        down_left = [cls.DOWN, cls.LEFT]
        up_right = [cls.UP, cls.RIGHT]
        random.shuffle(down_left)
        random.shuffle(up_right)
        for action in down_left + up_right:
            if action in valid_actions:
                return action
        assert False, "should be able to do something"

    down_left_fn.info = "Down Left strategy"
    do_trials(cls, down_left_fn)


def try_greedy(cls):
    def greedy_fn(board):
        test_game = cls()
        test_game.set_board(board)
        action_rewards = test_game.get_valid_actions_by_reward()
        assert len(action_rewards) > 0, "No actions available"
        return action_rewards[0][0]

    greedy_fn.info = "Greedy strategy"
    do_trials(cls, greedy_fn)


def try_down_left_greedy(cls):
    # pick best of {down, left} if available, otherwise best of {up, right}
    def down_left_greedy_fn(board):
        test_game = cls()
        test_game.set_board(board)
        action_rewards = test_game.get_valid_actions_by_reward()
        assert len(action_rewards) > 0, "No actions available"
        for action, reward in action_rewards:
            if action in [cls.DOWN, cls.LEFT]:
                return action
        return action_rewards[0][0]

    down_left_greedy_fn.info = "Down left greedy strategy"
    do_trials(cls, down_left_greedy_fn)


def do_stats():
    print(f"\nRunning {TRIALS} trials with Nick impl to test each strategy\n")
    try_only_go_right(Nick2048)
    try_random(Nick2048)
    try_down_left(Nick2048)
    try_greedy(Nick2048)
    try_down_left_greedy(Nick2048)
    print(f"\nRunning {TRIALS} trials with Andy impl to test each strategy\n")
    try_only_go_right(Andy2048)
    try_random(Andy2048)
    try_down_left(Andy2048)
    try_greedy(Andy2048)
    try_down_left_greedy(Andy2048)


if __name__ == "__main__":
    do_stats()
