import statistics
from strategies.utility import do_trials

ROLLOUT_DEPTH_LIMIT = 10
ROLLOUTS_PER_MOVE = 100


def best_next_move_from_random_rollouts(cls, board):
    action_rewards = cls.get_valid_actions_from_board(board)
    valid_actions = [a for (a, r, b) in action_rewards]
    test_game = cls()
    best_rollouts = {}
    for action in valid_actions:
        best_rollout_score = 0
        for i in range(ROLLOUTS_PER_MOVE):
            rollout = []
            test_game.score = 0
            test_game.set_board(board)
            test_game.step(action)
            rollout.append(action)
            depth = 0
            while not test_game.done and depth < ROLLOUT_DEPTH_LIMIT:
                old_board = test_game.board
                tmp_action = test_game.action_space.sample()
                test_game.step(tmp_action)
                if test_game.board != old_board:
                    rollout.append(tmp_action)
                    depth += 1
            if test_game.score >= best_rollout_score:
                best_rollout_score = test_game.score
                best_rollouts[action] = rollout
    avg_score = {}
    for action, rollout in best_rollouts.items():
        scores = []
        for i in range(ROLLOUTS_PER_MOVE):
            test_game.set_board(board)
            test_game.score = 0
            for a in rollout:
                test_game.step(a)
            scores.append(test_game.score)
        avg_score[action] = statistics.mean(scores)
    action = max(avg_score.items(), key=lambda x: x[1])[0]
    return action


def try_limited_tree_search(cls, trial_count):
    def rollout_fn(board):
        return best_next_move_from_random_rollouts(cls, board)

    rollout_fn.info = f"{ROLLOUTS_PER_MOVE} "
    rollout_fn.info += f"random rollouts per move "
    rollout_fn.info += f"with a depth limit of {ROLLOUT_DEPTH_LIMIT} "
    rollout_fn.info += f"per rollout."

    do_trials(cls, trial_count, rollout_fn, always_print=True)
