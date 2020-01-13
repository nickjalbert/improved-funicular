import time
import statistics


def do_trials(cls, trial_count, strategy, check_done_fn=None):
    start_time = time.time()
    scores = []
    max_tiles = []
    step_counts = []
    for i in range(trial_count):
        game = cls()
        curr_board, score, done = game.get_state()
        steps = 0
        while not done:
            assert curr_board == game.board
            prev_board = game.board
            move = strategy(curr_board)
            curr_board, reward, done = game.step(move)
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
