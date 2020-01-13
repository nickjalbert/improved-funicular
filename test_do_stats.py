from nick_2048 import Nick2048
from do_stats import STRATEGIES, IMPLEMENTATIONS
from strategies.random import try_random


def test_strategies():
    TO_TEST = [
        "only_go_right",
        "random",
        "down_left",
        "fixed_action_order",
        "greedy",
        "greedy_fixed_order",
        "down_left_greedy",
        "max_space_then_greedy",
        "lookahead_1",
        "lookahead_2",
    ]
    for strat_name in TO_TEST:
        STRATEGIES[strat_name](Nick2048, 2)


def test_implementations():
    for impl_name, implementation in IMPLEMENTATIONS.items():
        try_random(implementation, 2)
