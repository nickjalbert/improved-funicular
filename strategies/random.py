import random
from strategies.utility import do_trials


def try_random(cls, trial_count):
    def random_fn(board):
        choices = [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]
        return random.choice(choices)

    random_fn.info = "Random strategy"
    do_trials(cls, trial_count, random_fn)


try_random.info = "Try random moves until game over"
