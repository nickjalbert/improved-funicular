from strategies.utility import do_trials


def try_max_space_then_greedy(cls, trial_count):
    def max_space_then_greedy_fn(board):
        actions = cls.get_valid_actions_from_board(board)
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
    do_trials(cls, trial_count, max_space_then_greedy_fn)


try_max_space_then_greedy.info = (
    "Pick moves greedily with respect to free space on board, "
    "tiebreak by being greedy w.r.t. score"
)
