import random
from strategies.utility import do_trials


def try_greedy_fixed_order(cls, trial_count):
    ORDER = [cls.UP, cls.DOWN, cls.LEFT, cls.RIGHT]
    random.shuffle(ORDER)

    # Greedy, break ties in a fixed order
    def greedy_fixed_order_fn(board):
        actions = cls.get_valid_actions_by_reward_from_board(board)
        top_reward = actions[0][1]
        equiv_rewards = [a for (a, r, b) in actions if r >= top_reward]
        assert len(equiv_rewards) > 0, "No actions available"
        for action in ORDER:
            if action in equiv_rewards:
                return action
        assert False

    greedy_fixed_order_fn.info = "Greedy strategy with fixed preference"
    do_trials(cls, trial_count, greedy_fixed_order_fn)


try_greedy_fixed_order.info = (
    "Choose action that results in best score, fixed order tiebreak "
    "(e.g. Up > Down > Left > Right)"
)
