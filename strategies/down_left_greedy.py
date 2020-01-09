from strategies.utility import do_trials


def try_down_left_greedy(cls, trial_count):
    # pick best of {down, left} if available, otherwise best of {up, right}
    def down_left_greedy_fn(board):
        actions = cls.get_valid_actions_by_reward_from_board(board)
        assert len(actions) > 0, "No actions available"
        for action, reward, board in actions:
            if action in [cls.DOWN, cls.LEFT]:
                return action
        return actions[0][0]

    down_left_greedy_fn.info = "Down left greedy strategy"
    do_trials(cls, trial_count, down_left_greedy_fn)


try_down_left_greedy.info = (
    "Pick best of {down, left} if allowed, else best of {up, right}"
)
