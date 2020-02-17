import numpy as np
import time
import statistics
from collections import Counter


# From https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def print_results(strategy, elapsed, step_counts, max_tiles, scores):
    trial_count = len(step_counts)
    elapsed_per_trial = elapsed / trial_count
    elapsed = round(elapsed, 2)
    elapsed_per_trial = round(elapsed_per_trial, 5)
    mean_steps = round(statistics.mean(step_counts), 1)
    mean_step_time = round(elapsed / sum(step_counts), 5)
    max_tiles_str = ""
    max_tile_counts = Counter(max_tiles)
    for tile, count in sorted(max_tile_counts.most_common(), reverse=True):
        percentage = round(count * 100 / len(max_tiles), 2)
        max_tiles_str += f"\t\t{tile}: {count} ({percentage}%)\n"
    mean = statistics.mean(scores) if trial_count > 1 else scores[0]
    median = statistics.median(scores) if trial_count > 1 else scores[0]
    stdev = statistics.stdev(scores) if trial_count > 1 else 0
    print(
        f"{strategy.info} "
        f"({trial_count} trials, "
        f"{elapsed} sec total, {elapsed_per_trial} sec per trial):\n"
        f"\n\tMean steps per game: {mean_steps}\n"
        f"\tMean time per step: {mean_step_time}\n"
        f"\n\tMax Tiles:\n"
        f"{max_tiles_str}"
        f"\n\tMax Score: {max(scores)}\n"
        f"\tMean Score: {mean}\n"
        f"\tMedian Score: {median}\n"
        f"\tStandard Dev: {stdev}\n"
        f"\tMin Score: {min(scores)}\n"
    )


def do_trials(
    cls,
    trial_count,
    strategy,
    check_done_fn=None,
    max_steps_per_episode=500,
    random_seed=None,
    init_board=None,
    always_print=False,
):
    start_time = time.time()
    scores = []
    max_tiles = []
    step_counts = []
    for i in range(trial_count):
        game = cls(random_seed=random_seed)
        if init_board:
            game.set_board(init_board)
        curr_board, score, done = game.get_state()
        steps = 0
        while steps < max_steps_per_episode and not done:
            assert np.array_equal(np.asarray(curr_board), np.asarray(game.board))
            move = strategy(curr_board)
            prev_board = game.board
            curr_board, reward, done, _ = game.step(move)
            steps += 1
            if check_done_fn is not None:
                done = check_done_fn(prev_board, curr_board, reward, done)
        _, score, _ = game.get_state()
        scores.append(score)
        max_tiles.append(max(game.board))
        step_counts.append(steps)
        elapsed = time.time() - start_time
        if always_print:
            print_results(strategy, elapsed, step_counts, max_tiles, scores)
    print("\n=================\n")
    print("Final Results")
    print_results(strategy, elapsed, step_counts, max_tiles, scores)
    return {
        "Max Tile": max(max_tiles),
        "Max Score": max(scores),
        "Mean Score": statistics.mean(scores),
        "Median Score": statistics.median(scores),
        "Standard Dev": statistics.stdev(scores),
        "Min Score": min(scores),
    }
