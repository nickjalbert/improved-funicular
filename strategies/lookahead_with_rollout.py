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
            # tmp = cls()
            # tmp.board = board
            # tmp.render_board()
            return best_next_move_from_random_rollouts(cls, board)
            # print(action)

    lookahead_with_rollout_fn.info = f"Lookahead 3 with rollout strategy"
    do_trials(cls, trial_count, lookahead_with_rollout_fn)
