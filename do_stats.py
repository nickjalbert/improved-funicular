import time
import random
from collections import defaultdict
import numpy as np
import statistics
from nick_2048 import Nick2048
from andy_adapter import Andy2048

TRIALS = 100


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
        action_rewards = Nick2048.get_valid_actions_from_board(board)
        valid_actions = [a for (a, r, b) in action_rewards]
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


def try_fixed_action_order(cls):
    # Always choose actions in a particular order if they are valid
    def fixed_fn(board):
        ACTION_ORDER = [cls.DOWN, cls.LEFT, cls.UP, cls.RIGHT]
        actions = Nick2048.get_valid_actions_from_board(board)
        actions = [a for (a, r, b) in actions]
        assert len(actions) > 0, "No actions available"
        for action in ACTION_ORDER:
            if action in actions:
                return action
        assert False, "Could not find action"

    fixed_fn.info = "Fixed order strategy"
    do_trials(cls, fixed_fn)


def try_greedy(cls):
    # Greedy, break ties randomly
    def greedy_fn(board):
        actions = Nick2048.get_valid_actions_by_reward_from_board(board)
        assert len(actions) > 0, "No actions available"
        return actions[0][0]

    greedy_fn.info = "Greedy strategy"
    do_trials(cls, greedy_fn)


def try_greedy_fixed_order(cls):
    ORDER = [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]
    random.shuffle(ORDER)

    # Greedy, break ties in a fixed order
    def greedy_fixed_order_fn(board):
        actions = Nick2048.get_valid_actions_by_reward_from_board(board)
        top_reward = actions[0][1]
        equiv_rewards = [a for (a, r, b) in actions if r >= top_reward]
        assert len(equiv_rewards) > 0, "No actions available"
        for action in ORDER:
            if action in equiv_rewards:
                return action
        assert False

    greedy_fixed_order_fn.info = "Greedy strategy with fixed preference"
    do_trials(cls, greedy_fixed_order_fn)


def try_down_left_greedy(cls):
    # pick best of {down, left} if available, otherwise best of {up, right}
    def down_left_greedy_fn(board):
        actions = Nick2048.get_valid_actions_by_reward_from_board(board)
        assert len(actions) > 0, "No actions available"
        for action, reward, board in actions:
            if action in [cls.DOWN, cls.LEFT]:
                return action
        return actions[0][0]

    down_left_greedy_fn.info = "Down left greedy strategy"
    do_trials(cls, down_left_greedy_fn)


def try_max_space_then_greedy(cls):
    def max_space_then_greedy_fn(board):
        actions = Nick2048.get_valid_actions_from_board(board)
        assert len(actions) > 0, "No actions available"

        def key_fn(el):
            return (-1 * el[2].count(0), -1 * el[1])

        sorted_actions = sorted(actions, key=key_fn)
        top_action, top_reward, top_board = sorted_actions[0]
        top_action_zeros = top_board.count(0)
        for (a, r, b) in sorted_actions:
            assert b.count(0) <= top_action_zeros
        equiv_actions = [
            a
            for (a, r, b) in sorted_actions
            if b.count(0) >= top_action_zeros and r >= top_reward
        ]
        ORDER = [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]
        for action in ORDER:
            if action in equiv_actions:
                return action
        assert False

    max_space_then_greedy_fn.info = "Max space then greedy"
    do_trials(cls, max_space_then_greedy_fn)


def _get_lookahead_seqs(cls, action_space, lookahead_count):
    lookahead_seqs = [(cls.UP,), (cls.DOWN,), (cls.LEFT,), (cls.RIGHT,)]
    for i in range(lookahead_count - 1):
        new_lookahead_seqs = []
        for seq in lookahead_seqs:
            for action in action_space:
                new_lookahead_seqs.append((seq) + (action,))
        lookahead_seqs = new_lookahead_seqs
    return lookahead_seqs


def get_lookahead_fn(cls, lookahead_count):
    """Returns a function that takes a board and returns a suggested action."""
    action_space = [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]
    lookahead_seqs = _get_lookahead_seqs(cls, action_space, lookahead_count)

    # try each lookahead sequence, return the max action
    def lookahead_fn(board):
        action_rewards = cls.get_valid_actions_from_board(board)
        valid_actions = [a for (a, r, b) in action_rewards]
        test_board = cls()
        max_score = 0
        max_actions = set()
        for sequence in lookahead_seqs:
            if sequence[0] not in valid_actions:
                continue
            test_board.score = 0
            test_board.set_board(board)
            for action in sequence:
                test_board.step(action)
            if test_board.score > max_score:
                max_actions = set([sequence[0]])
                max_score = test_board.score
            if test_board.score == max_score:
                max_actions.add(sequence[0])
        for action in action_space:
            if action in max_actions:
                return action
        assert False

    return lookahead_fn


def try_lookahead(cls, lookahead_count):
    lookahead_fn = get_lookahead_fn(cls, lookahead_count)
    lookahead_fn.info = f"Lookahead {lookahead_count} strategy"
    do_trials(cls, lookahead_fn)


# globals so that we'll keep value function state across training rollouts
n = defaultdict(int)  # tuple(board), action -> count of visits
sum_ret = defaultdict(int)  # tuple(board), action -> expected return val


def train_mcts(cls, next_action, num_rollouts=100000):
    epsilon = 0.1
    discount_rate = 0.95
    print("training MCTS value function")
    for rollout_num in range(num_rollouts):
        if rollout_num % 2000 == 0:
            print("training rollout num %s" % rollout_num)
        game = cls()
        curr_board, score, done = game.get_state()
        states, actions, rewards = [], [], []
        while not done:
            assert curr_board == game.board
            if random.random() < epsilon:
                action = cls.random_direction()
            else:
                action = next_action(curr_board)
            states.append(curr_board)
            curr_board, reward, done = game.step(action)
            actions.append(action)
            rewards.append(reward)

        # calculate returns (i.e., discounted future rewards) using bellman backups for the rollout just completed
        returns = [0] * len(rewards)
        returns[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            n[(tuple(states[i]), actions[i])] += 1
            sum_ret[(tuple(states[i]), actions[i])] += (
                rewards[i] + discount_rate * returns[i + 1]
            )


def try_mcts(cls):
    action_space = cls().action_space

    def q(s, a):
        if n[(tuple(s), a)]:
            return sum_ret[(tuple(s), a)] / n[(tuple(s), a)]
        else:
            return 0

    def next_action(s):
        # return np.argmax([q(s, a) for a in action_space])
        # break ties with random coin flip.
        action_values = np.array([q(s, a) for a in action_space])
        return np.random.choice(np.flatnonzero(action_values == action_values.max()))

    def mcts_fn(board):
        return next_action(board)

    mcts_fn.info = "MCTS strategy"
    for _ in range(10):
        train_mcts(cls, next_action, 10000)
        do_trials(cls, mcts_fn)


def do_stats():
    print(f"\nRunning {TRIALS} trials with Nick impl to test each strategy\n")
    try_only_go_right(Nick2048)
    try_random(Nick2048)
    # try_down_left(Nick2048)
    # try_fixed_action_order(Nick2048)
    # try_greedy(Nick2048)
    # try_greedy_fixed_order(Nick2048)
    try_down_left_greedy(Nick2048)
    # try_max_space_then_greedy(Nick2048)
    try_lookahead(Nick2048, 3)
    try_mcts(Nick2048)
    print(f"\nRunning {TRIALS} trials with Andy impl to test each strategy\n")
    try_only_go_right(Andy2048)
    try_random(Andy2048)
    # try_down_left(Andy2048)
    # try_greedy(Andy2048)
    # try_down_left_greedy(Andy2048)


if __name__ == "__main__":
    do_stats()
