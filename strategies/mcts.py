import random
from collections import defaultdict
import numpy as np
import mlflow
from strategies.utility import do_trials, softmax

n = defaultdict(int)
sum_ret = defaultdict(lambda: 0.001)
num_revisits = 0


def train_tabular_mcts(
    cls,
    policy_fn,
    n,
    sum_ret,
    num_rollouts=100000,
    epsilon=0.9,
    discount_rate=0.95,
    perc_rollouts_full_random=10,
    rollout_start_count=0,
    init_board=None,
    print_stats=False,
    print_freq=100,
):
    global num_revisits
    for rollout_num in range(num_rollouts):
        rollout_num += rollout_start_count
        num_rollouts_full_random = num_rollouts * perc_rollouts_full_random / 100
        if rollout_num <= num_rollouts_full_random:
            prob_rand_action = 1
        else:
            prob_rand_action = min(
                1, 10 * epsilon / (rollout_num - num_rollouts_full_random + 1)
            )
        # print("prob of rand action: %s" % prob_rand_action)
        game = cls()
        if init_board:
            game.score = 0
            game.set_board(init_board)
        curr_board, score, done = game.get_state()
        states, actions, rewards, is_duplicate = [], [], [], []
        state_action_pairs = set()
        step_num = 0
        while not done:
            if random.random() < prob_rand_action:
                action = game.action_space.sample()
            else:
                action = policy_fn(curr_board)
            states.append(curr_board)
            actions.append(action)
            is_duplicate.append((tuple(curr_board), action) in state_action_pairs)
            state_action_pairs.add((tuple(curr_board), action))
            new_board, reward, done, _ = game.step(action)
            rewards.append(reward)

            # TODO: use a DNN as a value function, and use n-step Temporal Difference learning.

            curr_board = new_board
            step_num += 1

        # calculate returns (i.e., discounted future rewards) using bellman backups for the rollout just completed
        returns = [0] * len(rewards)
        returns[-1] = rewards[-1]
        num_revisits_this_rollout = 0
        for i in range(len(rewards) - 2, -1, -1):
            if (tuple(states[i]), actions[i]) in sum_ret:
                num_revisits_this_rollout += 1
            returns[i] = rewards[i] + discount_rate * returns[i + 1]
            if not is_duplicate[i]:
                n[(tuple(states[i]), actions[i])] += 1
                sum_ret[(tuple(states[i]), actions[i])] += returns[i]
        num_revisits += num_revisits_this_rollout

        stats = {
            "num_steps": len(rewards),
            "game_score": sum(rewards),
            "prob random action": prob_rand_action,
            "len sum_ret dict": len(sum_ret),
            "total num states-action pairs revisited": num_revisits,
            "percent state-action pairs in this rollout seen already": num_revisits_this_rollout
            / step_num
            * 100.0,
        }
        if print_stats and rollout_num % print_freq == 0:
            print("rollout num %s" % rollout_num)
            for k, v in stats.items():
                print("%s: %s" % (k, v))
            print()

        mlflow.log_metrics(stats, step=rollout_num)
    return num_revisits


def try_mcts(cls, trial_count):
    action_space = cls().action_space

    def q(s, a, n, sum_ret):
        if n[(tuple(s), a)]:
            return sum_ret[(tuple(s), a)] / n[(tuple(s), a)]
        else:
            return 0

    def mcts_policy_fn(board):
        # return action_space[np.argmax([q(s, a) for a in range(action_space.n)])]
        action_values = np.array(
            [q(board, a, n, sum_ret) for a in range(action_space.n)]
        )
        # return action_space[np.random.choice(np.flatnonzero(action_values == action_values.max()))] # break ties with random coin flip.
        action = np.argmax(np.random.multinomial(1, softmax(action_values)))
        return action

    def dont_repeat_done(prev, curr, reward, done):
        return done or prev == curr

    mcts_policy_fn.info = "MCTS strategy"
    with mlflow.start_run():
        epsilon = 1
        discount_rate = 0.8
        init_board = [2, 2] + [0] * 14
        num_training_iters = 10
        rollouts_per_training_iter = 1000
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("discount_rate", discount_rate)
        mlflow.log_param("init_board", str(init_board))
        mlflow.log_param("num_training_iter", num_training_iters)
        mlflow.log_param("rollouts_per_training_iter", rollouts_per_training_iter)
        for training_iter in range(num_training_iters):
            print("training MCTS policy for iteration %s" % training_iter)
            train_tabular_mcts(
                cls,
                mcts_policy_fn,
                n,
                sum_ret,
                epsilon=epsilon,
                discount_rate=discount_rate,
                num_rollouts=rollouts_per_training_iter,
                rollout_start_count=rollouts_per_training_iter * training_iter,
                init_board=init_board,
                print_stats=True,
            )
            print(
                "testing MCTS performance with %s trials for iteration %s"
                % (trial_count, training_iter)
            )
            trial_result = do_trials(
                cls, trial_count, mcts_policy_fn, init_board=init_board,
            )
            mlflow.log_metrics(trial_result, step=training_iter)


try_mcts.info = "Classic Monte Carlo tree search algorithm"
