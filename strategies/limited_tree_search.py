import statistics
import random
from strategies.utility import do_trials
from strategies.lookahead import get_lookahead_fn

LOOKAHEAD_COUNT = 3
ROLLOUT_THRESHOLD = LOOKAHEAD_COUNT + 4
ROLLOUTS_PER_MOVE = 150


def best_next_move_from_random_rollouts(cls, board):
    action_rewards = cls.get_valid_actions_from_board(board)
    valid_actions = [a for (a, r, b) in action_rewards]
    avg_steps = {}
    test_game = cls()
    for action in valid_actions:
        scores = []
        for i in range(ROLLOUTS_PER_MOVE):
            test_game.score = 0
            test_game.set_board(board)
            test_game.step(action)
            while not test_game.done:
                test_game.step(random.choice(test_game.action_space))
            scores.append(test_game.score)
        avg_steps[action] = statistics.mean(scores)
    action = max(avg_steps.items(), key=lambda x: x[1])[0]
    return action


def try_lookahead_with_rollout(cls, trial_count):
    lookahead_fn = get_lookahead_fn(cls, LOOKAHEAD_COUNT)

    def lookahead_with_rollout_fn(board):
        if len([v for v in board if v == 0]) > ROLLOUT_THRESHOLD:
            return lookahead_fn(board)
        else:
            return best_next_move_from_random_rollouts(cls, board)

    lookahead_with_rollout_fn.info = f"Lookahead {LOOKAHEAD_COUNT} with "
    lookahead_with_rollout_fn.info += f"{ROLLOUTS_PER_MOVE} "
    lookahead_with_rollout_fn.info += f"random rollouts per move "
    lookahead_with_rollout_fn.info += f"when board has <= {ROLLOUT_THRESHOLD} "
    lookahead_with_rollout_fn.info += f"empty spaces"
    do_trials(cls, trial_count, lookahead_with_rollout_fn, always_print=True)


def try_limited_tree_search(cls, trial_count):
    def rando_fn(board):
        return random.choice([0, 1, 2, 3])

    do_trials(cls, trial_count, rando_fn, always_print=True)
