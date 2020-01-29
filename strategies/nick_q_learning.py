# Q learning as presented on Sutton and Barto p131 (p153 in trimmed pdf)
from collections import defaultdict
import random
import time
import sys

from envs.nick_2048 import Nick2048
from strategies.utility import do_trials

ALPHA = 0.1
EPSILON = 0.1
DISCOUNT = 0.95


class QTable:
    def __init__(self):
        self.q_table = defaultdict(int)
        self.reset_counters()

    def get(self, state, action):
        self.lookups += 1
        if (state, action) in self.q_table:
            self.hits += 1
        val = self.q_table[(state, action)]
        if val != 0:
            self.nonzero_hits += 1
        return val

    def set(self, state, action, val):
        self.q_table[(state, action)] = val

    def learn(self, curr_state, action, reward, next_state):
        curr_q = self.get(curr_state, action)
        action_tuples = Nick2048.get_valid_actions_from_board(next_state)
        actions = [a for a, _, _ in action_tuples]
        if not actions:  # game is over
            return
        next_action_values = [(self.get(next_state, a), a) for a in actions]
        max_next_q, _ = max(next_action_values)
        q_update = ALPHA * (reward + DISCOUNT * max_next_q - curr_q)
        self.set(curr_state, action, curr_q + q_update)

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
    while not game.done:
        action = choose_action_epsilon_greedily(game, q_table)
        next_state, reward, _, _ = game.step(action)
        q_table.learn(curr_state, action, reward, next_state)
        curr_state = next_state


def choose_action_epsilon_greedily(game, q_table):
    random.seed(time.time())
    actions = [a for a, _, _ in game.get_valid_actions()]
    if random.random() < EPSILON:
        return random.choice(actions)
    else:
        action_values = [
            (q_table.get(game.board, action), action) for action in actions
        ]
        return max(action_values)[1]


def try_nick_q_learning(cls, trial_count):
    start = time.time()
    i = 0
    total_score = 0
    last_scores_to_store = 10
    last_x_scores = []
    game = Nick2048()
    q_table = QTable()
    while True:
        run_episode(game, q_table)
        total_score += game.score
        last_x_scores.append(game.score)
        while len(last_x_scores) > last_scores_to_store:
            last_x_scores.pop(0)
        i += 1
        if i % 100 == 0:
            total_sec = round(time.time() - start, 2)
            sec_per_iter = round(total_sec / i, 2)
            avg_game_score = round(total_score / i, 0)
            avg_last_x = round(sum(last_x_scores) / len(last_x_scores), 2)
            print(
                f"Training iteration {i} "
                f"({total_sec} sec total, {sec_per_iter} sec per iter)"
                f"\n\tLast {last_scores_to_store}: {last_x_scores} "
                f"(avg: {avg_last_x})"
                f"\n\tMean game score: {avg_game_score}"
                f"\n\tSize of state value table: "
                f"{round(q_table.size_in_mb, 2)}MB"
                f"\n\tState value hit rate: "
                f"{round(q_table.hit_rate, 2)}% "
                f"({q_table.lookup_hits} out of "
                f"{q_table.lookup_count})"
                f"\n\tState value non-zero hit rate: "
                f"{round(q_table.nonzero_hit_rate, 2)}% "
                f"({q_table.lookup_nonzero_hits} out of "
                f"{q_table.lookup_count})\n"
            )
            q_table.reset_counters()
        if i % 1000 == 0:
            q_table.reset_counters()

            def q_learning_benchmark_fn(board):
                action_tuples = Nick2048.get_valid_actions_from_board(board)
                actions = [a for a, _, _ in action_tuples]
                action_values = [(q_table.get(board, a), a) for a in actions]
                return max(action_values)[1]

            q_learning_benchmark_fn.info = f"Q-learning iteration {i}"
            do_trials(cls, trial_count, q_learning_benchmark_fn)
            print(
                f"State value hit rate: "
                f"{round(q_table.hit_rate, 2)}% "
                f"({q_table.lookup_hits} out of "
                f"{q_table.lookup_count})\n"
                f"State value non-zero hit rate: "
                f"{round(q_table.nonzero_hit_rate, 2)}% "
                f"({q_table.lookup_nonzero_hits} out of "
                f"{q_table.lookup_count})\n\n"
                f"=================\n\n"
            )
