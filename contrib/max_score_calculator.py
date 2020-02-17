from collections import deque
from envs.nick_2048 import Nick2048
import logging
import mlflow
import numpy as np
import time

# logging.basicConfig(level=logging.DEBUG)
start_time = time.time()
with mlflow.start_run():
    max_depth = 15
    assert max_depth > 0
    num_random_seeds = 100
    max_max_tile = []
    max_score = []
    total_state_action_pairs = []
    for rand_seed in range(num_random_seeds):
        max_max_tile.append([0] * (max_depth + 1))
        max_score.append([0] * (max_depth + 1))
        total_state_action_pairs.append([0] * (max_depth + 1))
        state_action_scores = {}
        env = Nick2048(random_seed=rand_seed)
        actions = range(env.action_space.n)
        state_actions = (
            deque()
        )  # queue of (depth, game_score, max_tile, state, next_action)

        init_state = env.get_state()[0]
        for a in actions:
            state_actions.append(
                (1, 0, max(init_state), init_state, a)
            )  # push initial actions

        while state_actions:
            debug_str = ""
            t = state_actions.popleft()
            debug_str += f"handling {t}\n"
            depth, game_score, max_tile, state, next_action = t
            if (state, next_action) in state_action_scores:
                if game_score <= state_action_scores[(state, next_action)]:
                    continue
            state_action_scores[(state, next_action)] = game_score
            env.set_board(state)
            env.score = game_score
            next_state, reward, done, _ = env.step(next_action)
            debug_str += (
                f"  next_state = {next_state}, reward = {reward}, done = {done}\n"
            )
            new_max = max(next_state)
            if new_max > max(max_max_tile[rand_seed]):
                debug_str += f"  new max_tile: {new_max}\n"
            max_max_tile[rand_seed][depth] = max(max(max_max_tile[rand_seed]), new_max)
            new_score = game_score + reward
            if new_score > max(max_score[rand_seed]):
                debug_str += f"  new max_score: {new_score}\n"
            max_score[rand_seed][depth] = max(max(max_score[rand_seed]), new_score)
            total_state_action_pairs[rand_seed][depth] += 1
            if depth < max_depth and not done:
                for a in actions:
                    if next_state == state and a == next_action:
                        debug_str += f"  not repeating a dud action {a}"
                    else:
                        state_actions.append(
                            (depth + 1, new_score, max_tile, next_state, a)
                        )
            logging.debug(debug_str)

    runtime = time.time() - start_time
    max_max_tile_dist = np.asarray(max_max_tile)
    max_score_dist = np.asarray(max_score)
    total_state_action_pairs_dist = np.asarray(total_state_action_pairs)
    for i in range(1, max_depth + 1):
        print(
            f"Depth: {i}\n"
            + f"Max max tile: {max_max_tile_dist.mean(axis=0)[i]} mean, "
            + f"{np.median(max_max_tile_dist, axis=0)[i]} med, "
            + f"{max_max_tile_dist.std(axis=0)[i]} std, "
            + f"{max_max_tile_dist.max(axis=0)[i]} max\n"
            + f"Max score: {max_score_dist.mean(axis=0)[i]} mean, "
            + f"{np.median(max_score_dist, axis=0)[i]} med, "
            + f"{max_score_dist.std(axis=0)[i]} std, "
            + f"{max_score_dist.max(axis=0)[i]} max\n"
            + f"total_state_action_pairs: {total_state_action_pairs_dist.mean(axis=0)[i]} mean, "
            + f"{np.median(total_state_action_pairs_dist, axis=0)[i]} med, "
            + f"{total_state_action_pairs_dist.std(axis=0)[i]} std, "
            + f"{total_state_action_pairs_dist.max(axis=0)[i]} max\n"
        )
        # mlflow.log_metrics(
        #     {
        #         "Depth": i,
        #         "Max max tile": max_max_tile[rand_seed][i],
        #         "Max score": max_score[rand_seed][i],
        #         "total_state_action_pairs": total_state_action_pairs[rand_seed][i],
        #     },
        #     step=max_depth,
        # )
    print("runtime: %.3f seconds" % runtime)
    print("num random seeds: %s" % num_random_seeds)
