from strategies.utility import do_trials


def try_fixed_action_order(cls, trial_count):
    # Always choose actions in a particular order if they are valid
    def fixed_fn(board):
        ACTION_ORDER = [cls.DOWN, cls.LEFT, cls.UP, cls.RIGHT]
        actions = cls.get_valid_actions_from_board(board)
        actions = [a for (a, r, b) in actions]
        assert len(actions) > 0, "No actions available"
        for action in ACTION_ORDER:
            if action in actions:
                return action
        assert False, "Could not find action"

    fixed_fn.info = "Fixed order strategy"
    do_trials(cls, trial_count, fixed_fn)


try_fixed_action_order.info = (
    "Always choose actions in a fixed order if they are valid "
    "(Down > Left > Up > Right)"
)
