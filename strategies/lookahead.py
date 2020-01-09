from strategies.utility import do_trials


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


def try_lookahead(cls, trial_count, lookahead_count):
    lookahead_fn = get_lookahead_fn(cls, lookahead_count)
    lookahead_fn.info = f"Lookahead {lookahead_count} strategy"
    do_trials(cls, trial_count, lookahead_fn)
