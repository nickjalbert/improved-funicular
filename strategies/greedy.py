from strategies.utility import do_trials


def try_greedy(cls, trial_count):
    # Greedy, break ties randomly
    def greedy_fn(board):
        actions = cls.get_valid_actions_by_reward_from_board(board)
        assert len(actions) > 0, "No actions available"
        return actions[0][0]

    greedy_fn.info = "Greedy strategy"
    do_trials(cls, trial_count, greedy_fn)


try_greedy.info = "Choose action that results in best score, tiebreak randomly"
