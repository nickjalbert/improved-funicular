from strategies.utility import do_trials


def try_only_go_right(cls, trial_count):
    def right_fn(board):
        return cls.RIGHT

    def right_done(prev, curr, reward, done):
        return done or prev == curr

    right_fn.info = "Strategy only moves right"
    do_trials(cls, trial_count, right_fn, right_done)


try_only_go_right.info = "Only go right, and end when board stops changing"
