import numpy as np
import time
import statistics


# From https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def do_trials(cls, trial_count, strategy, check_done_fn=None, init_board=None):
    start_time = time.time()
    scores = []
    max_tiles = []
    step_counts = []
    for i in range(trial_count):
        game = cls()
        if init_board:
            game.set_board(init_board)
        curr_board, score, done = game.get_state()
        steps = 0
        while not done:
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
    elapsed_per_trial = elapsed / trial_count
    elapsed = round(elapsed, 2)
    elapsed_per_trial = round(elapsed_per_trial, 5)
    mean_steps = round(statistics.mean(step_counts), 1)
    mean_step_time = round(elapsed / sum(step_counts), 5)
    print(
        f"{strategy.info} "
        f"({elapsed} sec total, {elapsed_per_trial} sec per trial):\n"
        f"\n\tMean steps per game: {mean_steps}\n"
        f"\tMean time per step: {mean_step_time}\n"
        f"\tMax Tile: {max(max_tiles)}\n"
        f"\n\tMax Score: {max(scores)}\n"
        f"\tMean Score: {statistics.mean(scores)}\n"
        f"\tMedian Score: {statistics.median(scores)}\n"
        f"\tStandard Dev: {statistics.stdev(scores)}\n"
        f"\tMin Score: {min(scores)}\n"
    )
    return {"Max Tile": max(max_tiles),
            "Max Score": max(scores),
            "Mean Score": statistics.mean(scores),
            "Median Score": statistics.median(scores),
            "Standard Dev": statistics.stdev(scores),
            "Min Score": min(scores)}
