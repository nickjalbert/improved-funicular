from collections import deque
from envs.nick_2048 import Nick2048
import logging
import mlflow

# logging.basicConfig(level=logging.DEBUG)

with mlflow.start_run():
    max_depth = 10
    assert max_depth > 0
    env = Nick2048(random_seed=42)
    actions = range(env.action_space.n)
    state_actions = (
        deque()
    )  # queue of (depth, game_score, max_tile, state, next_action)
    max_max_tile = [0] * (max_depth + 1)
    max_score = [0] * (max_depth + 1)
    total_state_action_pairs = [0] * (max_depth + 1)

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
        env.set_board(state)
        env.score = game_score
        next_state, reward, done, _ = env.step(next_action)
        debug_str += f"  next_state = {next_state}, reward = {reward}, done = {done}\n"
        new_max = max(next_state)
        if new_max > max(max_max_tile):
            debug_str += f"  new max_tile: {new_max}\n"
        max_max_tile[depth] = new_max
        new_score = game_score + reward
        if new_score > max(max_score):
            debug_str += f"  new max_score: {new_score}\n"
        max_score[depth] = new_score
        total_state_action_pairs[depth] += 1
        if depth < max_depth and not done:
            for a in actions:
                if next_state == state and a == next_action:
                    debug_str += f"  not repeating a dud action {a}"
                else:
                    state_actions.append(
                        (depth + 1, new_score, max_tile, next_state, a)
                    )
        logging.debug(debug_str)

    for i in range(1, max_depth + 1):
        print(
            f"\nDepth: {i}\nMax max tile: {max_max_tile[i]}\nMax score: {max_score[i]}\n" +
            "total_state_action_pairs: {total_state_action_pairs[i]}\n"
        )
        mlflow.log_metrics(
            {
                "Depth": i,
                "Max max tile": max_max_tile[i],
                "Max score": max_score[i],
                "total_state_action_pairs": total_state_action_pairs[i],
            },
            step=max_depth,
        )
