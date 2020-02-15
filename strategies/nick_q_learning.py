# Q-learning (off-policy TD control) as presented in
# Sutton and Barto p131 (p153 in trimmed pdf)
# Run on 2048:
#   python do_stats.py nick 100 nick_q_learning
# Run on cartpole:
#   python do_stats.py nick 100 nick_q_learning_cartpole
from collections import defaultdict
import random
import time
import sys

import mlflow

from envs.nick_cartpole_adapter import NickCartpoleAdapter
from strategies.utility import do_trials

ALPHA = 0.1
EPSILON = 0.1
DISCOUNT = 0.95
DEPTH_LIMIT = 9
RANDOM_SEED = 42


class QTable:
    def __init__(self, test_cls):
        self.test_cls = test_cls
        self.q_table = defaultdict(int)
        self.reset_counters()

    def get_max_action(self, board):
        test_game = self.test_cls()
        test_game.set_board(board)
        actions = [a for a, _, _ in test_game.get_valid_actions()]
        if not actions:
            return None
        action_values = []
        for action in actions:
            canonical = self.test_cls.get_canonical_afterstate(board, action)
            val = self.get(canonical, action)
            action_values.append((val, action))
        return max(action_values)[1]

    def get(self, state, action):
        assert len(state) == self.test_cls.STATE_LEN
        assert action in self.test_cls.action_space
        self.lookups += 1
        if (state, action) in self.q_table:
            self.hits += 1
        val = self.q_table[(state, action)]
        if val != 0:
            self.nonzero_hits += 1
        return val

    def set(self, state, action, val):
        assert len(state) == self.test_cls.STATE_LEN
        assert action in self.test_cls.action_space
        self.q_table[(state, action)] = val

    def learn(self, curr_state, action, reward, next_state):
        curr_canonical = self.test_cls.get_canonical_afterstate(curr_state, action)
        curr_q = self.get(curr_canonical, action)
        max_next_action = self.get_max_action(next_state)
        if max_next_action is None:  # game is done
            return
        next_canonical = self.test_cls.get_canonical_afterstate(
            next_state, max_next_action
        )
        max_next_q = self.get(next_canonical, max_next_action)
        q_update = ALPHA * (reward + (DISCOUNT * max_next_q) - curr_q)
        self.set(curr_canonical, action, curr_q + q_update)

    def reset_counters(self):
        self.lookups = 0
        self.hits = 0
        self.nonzero_hits = 0

    @property
    def size_in_mb(self):
        return sys.getsizeof(self.q_table) / 2 ** 20

    @property
    def lookup_count(self):
        return self.lookups

    @property
    def lookup_hits(self):
        return self.hits

    @property
    def lookup_nonzero_hits(self):
        return self.nonzero_hits

    @property
    def hit_rate(self):
        if self.lookups == 0:
            return 0
        return self.hits * 100 / self.lookups

    @property
    def nonzero_hit_rate(self):
        if self.lookups == 0:
            return 0
        return self.nonzero_hits * 100 / self.lookups


# See Q-Learning pseudocode on Sutton & Barto p.131
def run_episode(game, q_table):
    game.reset()
    curr_state = game.board
    depth = 0
    while not game.done:
        if DEPTH_LIMIT and depth > DEPTH_LIMIT:
            break
        action = choose_action_epsilon_greedily(game, q_table)
        next_state, reward, _, _ = game.step(action)
        q_table.learn(curr_state, action, reward, next_state)
        curr_state = next_state
        depth += 1


def choose_action_epsilon_greedily(game, q_table):
    random.seed(time.time())
    if random.random() < EPSILON:
        actions = [a for a, _, _ in game.get_valid_actions()]
        return random.choice(actions)
    else:
        return q_table.get_max_action(game.board)


def _try_nick_q_learning(cls, trial_count):
    if RANDOM_SEED:
        print(f"RUNNING WITH RANDOM SEED {RANDOM_SEED}")
    if DEPTH_LIMIT:
        print(f"RUNNING WITH DEPTH LIMIT {DEPTH_LIMIT}")
    start = time.time()
    i = 0
    last_scores_to_store = 10
    all_scores = []
    game = cls(random_seed=RANDOM_SEED)
    q_table = QTable(cls)
    while True:
        run_episode(game, q_table)
        all_scores.append(game.score)
        i += 1
        if i % 100 == 0:
            total_sec = round(time.time() - start, 2)
            sec_per_iter = round(total_sec / i, 2)
            max_game_score = round(max(all_scores), 0)
            mean_game_score = round(sum(all_scores) / len(all_scores), 0)
            last_score_idx = -1 * last_scores_to_store
            last_x_scores = all_scores[last_score_idx:]
            avg_last_x = round(sum(last_x_scores) / len(last_x_scores), 2)
            print(
                f"Training iteration {i} "
                f"({total_sec} sec total, {sec_per_iter} sec per iter)"
                f"\n\tLast {last_scores_to_store}: {last_x_scores} "
                f"(avg: {avg_last_x})"
                f"\n\tMax game score: {max_game_score}"
                f"\n\tMean game score: {mean_game_score}"
                f"\n\tSize of state value table: "
                f"{round(q_table.size_in_mb, 2)}MB"
                f"\n\tQ table hit rate: "
                f"{round(q_table.hit_rate, 2)}% "
                f"({q_table.lookup_hits} out of "
                f"{q_table.lookup_count})"
                f"\n\tQ table non-zero hit rate: "
                f"{round(q_table.nonzero_hit_rate, 2)}% "
                f"({q_table.lookup_nonzero_hits} out of "
                f"{q_table.lookup_count})\n"
            )
            all_scores = []
            q_table.reset_counters()
        if i % 1000 == 0:
            q_table.reset_counters()

            def q_learning_benchmark_fn(board):
                return q_table.get_max_action(board)

            q_learning_benchmark_fn.info = f"Q-learning iteration {i}"
            results = do_trials(cls, trial_count, q_learning_benchmark_fn)
            mlflow.log_metric("max tile", results["Max Tile"], step=i)
            mlflow.log_metric("max score", results["Max Score"], step=i)
            mlflow.log_metric("mean score", results["Mean Score"], step=i)
            mlflow.log_metric("median score", results["Median Score"], step=i)
            mlflow.log_metric("stdev", results["Standard Dev"], step=i)
            mlflow.log_metric("min score", results["Min Score"], step=i)
            mlflow.log_metric("q hit rate", q_table.hit_rate, step=i)
            mlflow.log_metric("q nonzero hit rate", q_table.nonzero_hit_rate, step=i)
            mlflow.log_metric("q size", q_table.size_in_mb, step=1)

            print(
                f"Q table hit rate: "
                f"{round(q_table.hit_rate, 2)}% "
                f"({q_table.lookup_hits} out of "
                f"{q_table.lookup_count})\n"
                f"Q table non-zero hit rate: "
                f"{round(q_table.nonzero_hit_rate, 2)}% "
                f"({q_table.lookup_nonzero_hits} out of "
                f"{q_table.lookup_count})\n"
                f"Size of state value table: "
                f"{round(q_table.size_in_mb, 2)}MB\n\n"
                f"=================\n\n"
            )


def try_nick_q_learning(cls, trial_count):
    with mlflow.start_run():
        mlflow.log_params(
            {
                "alpha": ALPHA,
                "epsilon": EPSILON,
                "discount rate": DISCOUNT,
                "depth limit": DEPTH_LIMIT,
                "random seed": RANDOM_SEED,
                "desc": "q learning on 2048",
            }
        )
        return _try_nick_q_learning(cls, trial_count)


def try_nick_q_learning_cartpole(cls, trial_count):
    with mlflow.start_run():
        mlflow.log_params(
            {
                "alpha": ALPHA,
                "epsilon": EPSILON,
                "discount rate": DISCOUNT,
                "depth limit": DEPTH_LIMIT,
                "random seed": RANDOM_SEED,
                "desc": "q learning on CARTPOLE w/ canonical afterstates",
            }
        )
        return _try_nick_q_learning(NickCartpoleAdapter, trial_count)
