from nick_2048 import Nick2048
from do_stats import STRATEGIES, IMPLEMENTATIONS
from strategies.random import try_random


def test_strategies():
    SKIP = ["lookahead_3", "lookahead_4", "lookahead_5", "mcts"]
    for strat_name, strategy in STRATEGIES.items():
        if strat_name in SKIP:
            continue
        strategy(Nick2048, 2)


def test_implementations():
    for impl_name, implementation in IMPLEMENTATIONS.items():
        try_random(implementation, 2)
