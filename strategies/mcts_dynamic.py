import random
from collections import defaultdict
import numpy as np
import mlflow
from strategies.utility import do_trials


def train_mcts_dynamic(cls,
                       policy_fn,
                       n,
                       sum_ret,
                       num_rollouts=100000,
                       epsilon=0.9,
                       perc_rollouts_full_random=10,
                       discount_rate=0.95,
                       init_board=None):
    num_revisits = 0
    for rollout_num in range(num_rollouts):
        num_rollouts_full_random = num_rollouts * perc_rollouts_full_random / 100
        if rollout_num <= num_rollouts_full_random:
            prob_rand_action = 1
        else:
            prob_rand_action = min(1, 10 * epsilon / (rollout_num-num_rollouts_full_random + 1))
        #print("prob of rand action: %s" % prob_rand_action)
        if init_board:
            game = cls(init_board=init_board, init_score=0)
        else:
            game = cls()
        curr_board, score, done = game.get_state()
        states, actions, rewards, is_duplicate = [], [], [], []
        state_action_pairs = set()
        step_num = 0
        while not done:
            assert curr_board == game.board
            if random.random() < prob_rand_action:
                action = cls.random_direction()
            else:
                action = policy_fn(curr_board, n, sum_ret)
            states.append(curr_board.copy())
            actions.append(action)
            is_duplicate.append((tuple(curr_board), action) in state_action_pairs)
            state_action_pairs.add((tuple(curr_board), action))
            curr_board, reward, done = game.step(action)
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

        if rollout_num % 50 == 0:
            #print("== End of training rollout %s ==" % rollout_num)
            #print("epsilon based prob: %0.2f" % prob_rand_action)
            #print("len of sum_ret dict: %s" % len(sum_ret))
            #print("total num states-action pairs revisited: %s" % num_revisits)
            #print("%% state-action pairs in this rollout seen already: %.2f%%" % (num_revisits_this_rollout / step_num * 100.0))

            mlflow.log_metric("num_steps", len(rewards), step=rollout_num)
            mlflow.log_metric("game_score", sum(rewards), step=rollout_num)
            mlflow.log_metric("prob random action", prob_rand_action, step=rollout_num)
            mlflow.log_metric("len sum_ret dict", len(sum_ret), step=rollout_num)
            mlflow.log_metric("total num states-action pairs revisited", num_revisits, step=rollout_num)
            mlflow.log_metric("percent state-action pairs in this rollout seen already",
                              num_revisits_this_rollout / step_num * 100.0,
                              step=rollout_num)

# From https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def get_strategy_function(cls, epsilon, rollouts_per_move):
    action_space = cls().action_space

    def policy_fn(s, n, sum_ret):
        def q(a): # s is implicit since this function is nested inside of next_action.
            if n[(tuple(s), a)]:
                return sum_ret[(tuple(s), a)] / n[(tuple(s), a)]
            else:
                return 0
        action_values = np.array([q(a) for a in action_space])
        max_vals = np.flatnonzero(action_values == action_values.max())
        action = action_space[np.random.choice(max_vals)]
        return action  # break ties with random coin flip.
        #return np.argmax(np.random.multinomial(1, softmax(action_values)))

    def strategy_fn(board):
        n = defaultdict(int)  # tuple(board), action -> count of visits
        sum_ret = defaultdict(lambda: 0.001)  # tuple(board), action -> expected return val
        train_mcts_dynamic(cls, policy_fn, n, sum_ret,
                           epsilon=epsilon,
                           perc_rollouts_full_random=10,
                           discount_rate=0.2,
                           num_rollouts=rollouts_per_move,
                           init_board=board)
        print("finished training our value function. Here are the results for 4 actions given our current state:")
        for i in range(4):
            print(f"n(board, %s): %s" % (["R","D","L","U"][i], n[(tuple(board), i)]))
            print(f"sum_ret: %s" % sum_ret[(tuple(board), i)])
            if n[(tuple(board), i)]:
                avg_sum_ret = (sum_ret[(tuple(board), i)] / n[(tuple(board), i)])
            else:
                avg_sum_ret = 0
            print(f"avg sum_ret: %s" % avg_sum_ret)
            print("--")
        return policy_fn(board, n, sum_ret)

    return strategy_fn

def try_mcts_dynamic(cls, trial_count):
    with mlflow.start_run():
        epsilon = 1
        rollouts_per_move = 500
        mlflow.log_param("epsilon", epsilon)
        mlflow.log_param("rollouts_per_move", rollouts_per_move)
        strategy_fn = get_strategy_function(cls, epsilon, rollouts_per_move)
        strategy_fn.info = "MCTS_dynamic strategy"

        trial_result = do_trials(cls, trial_count, strategy_fn)
        mlflow.log_metrics(trial_result)
