# Temporal difference learning coded
from collections import defaultdict
import random
import time
import sys

# import tensorflow.keras as keras
# import tensorflow as tf
# import tensorflow_probability as tfp
# import mlflow
# import numpy as np
from envs.nick_2048 import Nick2048
from strategies.utility import do_trials

alpha = 0.1
epsilon = 0.1
discount_rate = 0.95
game = Nick2048()


class StateValue:
    def __init__(self):
        self.state_value = defaultdict(int)
        self.reset_counters()

    def get(self, state):
        canonical = Nick2048.get_canonical_board(state)
        self.lookups += 1
        if canonical in self.state_value:
            self.hits += 1
        val = self.state_value[canonical]
        if val != 0:
            self.nonzero_hits += 1
        return val

    def set(self, state, val):
        canonical = Nick2048.get_canonical_board(state)
        self.state_value[canonical] = val

    def reset_counters(self):
        self.lookups = 0
        self.hits = 0
        self.nonzero_hits = 0

    @property
    def size_in_mb(self):
        return sys.getsizeof(self.state_value) / 2 ** 20

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


state_value = StateValue()


def td_learn():
    game.reset()
    while not game.done:
        if random.random() <= epsilon:
            take_random_step()
        else:
            take_best_value_step()


def take_random_step():
    action, _, _ = random.choice(game.get_valid_actions())
    afterstate = game.get_afterstate(game.board, action)
    canonical_prev_board = game.get_canonical_board(afterstate)
    new_board, reward, is_done, info = game.step(action)
    canonical_new_board = game.get_canonical_board(new_board)
    do_td_update(canonical_prev_board, reward, canonical_new_board)


def get_max_action():
    value_actions = []
    for (action, _, _) in game.get_valid_actions():
        board = game.get_afterstate(game.board, action)
        discounted_reward = state_value.get(board)
        value_actions.append((discounted_reward, action, board))
    random.shuffle(value_actions)
    return max(value_actions)


def take_best_value_step(learn=True):
    max_discounted_reward, max_action, max_board = get_max_action()
    new_board, reward, is_done, info = game.step(max_action)
    if not game.done:
        _, _, new_board = get_max_action()
    if learn:
        do_td_update(max_board, reward, new_board)
    return max_action


def do_td_update(prev_state, reward, curr_state):
    prev_discounted_reward = state_value.get(prev_state)
    curr_discounted_reward = discount_rate * state_value.get(curr_state)
    td_difference = reward + curr_discounted_reward - prev_discounted_reward
    updated_discounted_reward = prev_discounted_reward + alpha * td_difference
    state_value.set(prev_state, updated_discounted_reward)


def td_fn(board):
    game.set_board(board)
    return take_best_value_step(learn=False)


def try_nick_td(cls, trial_count):
    i = 0
    start = time.time()
    total_score = 0
    last_scores_to_store = 10
    last_x_scores = []
    while True:
        td_learn()
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
                f"{round(state_value.size_in_mb, 2)}MB"
                f"\n\tState value hit rate: "
                f"{round(state_value.hit_rate, 2)}% "
                f"({state_value.lookup_hits} out of "
                f"{state_value.lookup_count})"
                f"\n\tState value non-zero hit rate: "
                f"{round(state_value.nonzero_hit_rate, 2)}% "
                f"({state_value.lookup_nonzero_hits} out of "
                f"{state_value.lookup_count})\n"
            )
            state_value.reset_counters()
        if i % 1000 == 0:
            state_value.reset_counters()
            td_fn.info = f"TD-0 learning iteration {i}"
            do_trials(cls, trial_count, td_fn)
            print(
                f"State value hit rate: "
                f"{round(state_value.hit_rate, 2)}% "
                f"({state_value.lookup_hits} out of "
                f"{state_value.lookup_count})\n"
                f"State value non-zero hit rate: "
                f"{round(state_value.nonzero_hit_rate, 2)}% "
                f"({state_value.lookup_nonzero_hits} out of "
                f"{state_value.lookup_count})\n\n"
                f"=================\n\n"
            )


def learn_with_keras():
    pass
    # model = keras.Sequential([
    #    keras.layers.Dense(256, activation='relu', input_dim=16),
    #    keras.layers.Dense(16, activation='relu'),
    #    keras.layers.Dense(1, activation='linear')
    # ])
    # model.summary()
    # board = (0,2,2,2, 0,0,4,2, 0,8,4,4, 0,0,0,0)
    # arr = np.reshape(board, (1,16))
    # print(model(arr))
