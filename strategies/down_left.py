import random
from strategies.utility import do_trials


def try_down_left(cls, trial_count):
    def down_left_fn(board):
        action_rewards = cls.get_valid_actions_from_board(board)
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
    do_trials(cls, trial_count, down_left_fn)


try_down_left.info = (
    "Only go down or left " "until you get stuck, then randomly go up or right once"
)
