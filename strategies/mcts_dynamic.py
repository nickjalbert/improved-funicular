import random
from collections import defaultdict
import numpy as np
import mlflow
from strategies.utility import do_trials, softmax


def train_tabular_mcts(
    cls,
    policy_fn,
    n,
    sum_ret,
    num_rollouts=100000,
    max_rollout_steps=20,
    epsilon=0.9,
    discount_rate=0.95,
    perc_rollouts_full_random=10,
    rollout_start_count=0,
    init_board=None,
    print_stats=False,
    print_freq=100,
):
    num_revisits = 0
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
        if init_board is not None:
            game.score = 0
            game.set_board(init_board)
        curr_board, score, done = game.get_state()
        curr_board = np.asarray(curr_board)
        states, actions, rewards, is_duplicate = [], [], [], []
        state_action_pairs = set()
        step_num = 0
        while not done and step_num < max_rollout_steps:
            assert np.array_equal(curr_board, np.asarray(game.board)), (
                curr_board,
                np.asarray(game.board),
            )
            if random.random() < prob_rand_action:
                action = cls.random_direction()
            else:
                action = policy_fn(curr_board, n, sum_ret)
            states.append(curr_board)
            actions.append(action)
            is_duplicate.append((tuple(curr_board), action) in state_action_pairs)
            state_action_pairs.add((tuple(curr_board), action))
            curr_board, reward, done, _ = game.step(action)
            curr_board = np.asarray(curr_board)
            print(["R", "D", "L", "U"][action])
            print()
            print(np.asarray(curr_board).reshape(4, 4))
            print()
            rewards.append(reward)
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

        # mlflow.log_metrics(stats, step=rollout_num)
    return num_revisits


def get_strategy_function(cls, epsilon, rollouts_per_move):
    action_space = cls().action_space

    def policy_fn(s, n, sum_ret):
        def q(a):  # s is implicit since this function is nested inside of next_action.
            if n[(tuple(s), a)]:
                return sum_ret[(tuple(s), a)] / n[(tuple(s), a)]
            else:
                return 0

        action_values = np.array([q(a) for a in range(action_space.n)])
        return np.random.choice(
            np.flatnonzero(action_values == action_values.max())
        )  # break ties with random coin flip.
        # return np.argmax(np.random.multinomial(1, softmax(action_values)))

    def strategy_fn(board):
        n = defaultdict(int)  # tuple(board), action -> count of visits
        sum_ret = defaultdict(
            lambda: 0.001
        )  # tuple(board), action -> expected return val
        train_tabular_mcts(
            cls,
            policy_fn,
            n,
            sum_ret,
            num_rollouts=rollouts_per_move,
            max_rollout_steps=5,
            epsilon=epsilon,
            perc_rollouts_full_random=10,
            discount_rate=0.2,
            init_board=board,
            print_stats=False,
        )
        # print("finished training our value function. Here are the results for 4 actions given our current state:")
        for i in range(4):
            # print(f"n(board, %s): %s" % (["R","D","L","U"][i], n[(tuple(board), i)]))
            # print(f"sum_ret: %s" % sum_ret[(tuple(board), i)])
            if n[(tuple(board), i)]:
                avg_sum_ret = sum_ret[(tuple(board), i)] / n[(tuple(board), i)]
            else:
                avg_sum_ret = 0
            # print(f"avg sum_ret: %s" % avg_sum_ret)
            # print("--")
        return policy_fn(board, n, sum_ret)

    return strategy_fn


def try_mcts_dynamic(cls, trial_count):
    with mlflow.start_run():
        epsilon = 1
        rollouts_per_move = 100
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("rollouts_per_move", rollouts_per_move)
        strategy_fn = get_strategy_function(cls, epsilon, rollouts_per_move)
        strategy_fn.info = "MCTS_dynamic strategy"

        trial_result = do_trials(cls, trial_count, strategy_fn)
        mlflow.log_metrics(trial_result)
