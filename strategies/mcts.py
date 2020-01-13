import random
from collections import defaultdict
import numpy as np
from strategies.utility import do_trials

# globals so that we'll keep value function state across training rollouts
n = defaultdict(int)  # tuple(board), action -> count of visits
sum_ret = defaultdict(lambda: 1)  # tuple(board), action -> expected return val


def train_mcts(cls, next_action, num_rollouts=100000, epsilon=0.1, discount_rate=0.95):
    print("training MCTS value function")
    for rollout_num in range(num_rollouts):
        prob_rand_action = min(1, 1000 * epsilon / (rollout_num + 1))
        if rollout_num % 2000 == 0:
            print("training rollout num %s" % rollout_num)
            print("epsilon: %0.2f" % prob_rand_action)
            print("len of sum_ret dict: %s" % len(sum_ret))
        game = cls()
        curr_board, score, done = game.get_state()
        states, actions, rewards, is_duplicate = [], [], [], []
        state_action_pairs = set()
        while not done:
            assert curr_board == game.board
            if random.random() < prob_rand_action:
                action = cls.random_direction()
            else:
                action = next_action(curr_board)
            states.append(curr_board)
            actions.append(action)
            is_duplicate.append((tuple(curr_board), action) in state_action_pairs)
            curr_board, reward, done = game.step(action)
            rewards.append(reward)

        # calculate returns (i.e., discounted future rewards) using bellman backups for the rollout just completed
        returns = [0] * len(rewards)
        returns[-1] = rewards[-1]
        for i in range(len(rewards) - 2, -1, -1):
            returns[i] = rewards[i] + discount_rate * returns[i + 1]
            if not is_duplicate[i]:
                n[(tuple(states[i]), actions[i])] += 1
                sum_ret[(tuple(states[i]), actions[i])] += returns[i]


def try_mcts(cls, trial_count):
    action_space = cls().action_space

    def q(s, a):
        if n[(tuple(s), a)]:
            return sum_ret[(tuple(s), a)] / n[(tuple(s), a)]
        else:
            return 0

    def next_action(s):
        # return np.argmax([q(s, a) for a in action_space])
        # break ties with random coin flip.
        action_values = np.array([q(s, a) for a in action_space])
        return np.random.choice(np.flatnonzero(action_values == action_values.max()))

    def mcts_fn(board):
        return next_action(board)

    mcts_fn.info = "MCTS strategy"
    for _ in range(10):
        train_mcts(cls, next_action, 10000)
        do_trials(cls, trial_count, mcts_fn)


try_mcts.info = "Classic Monte Carlo tree search algorithm"
