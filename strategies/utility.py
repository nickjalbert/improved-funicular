import time
import statistics


def do_trials(cls, trial_count, strategy, check_done_fn=None, init_board=None):
    start_time = time.time()
    scores = []
    max_tiles = []
    for i in range(trial_count):
        if init_board:
            game = cls(init_board.copy())
        else:
            game = cls()
        curr_board, score, done = game.get_state()
        while not done:
            assert curr_board == game.board
            move = strategy(curr_board)
            prev_board = curr_board[:]
            curr_board, reward, done = game.step(move)
            if check_done_fn is not None:
                done = check_done_fn(prev_board, curr_board, reward, done)
        _, score, _ = game.get_state()
        scores.append(score)
        max_tiles.append(max(game.board))
    elapsed = time.time() - start_time
    elapsed_per_trial = elapsed / trial_count
    elapsed = round(elapsed, 2)
    elapsed_per_trial = round(elapsed_per_trial, 5)
    print(
        f"{strategy.info} "
        f"({elapsed} sec total, {elapsed_per_trial} sec per trial):\n"
        f"\tMax Tile: {max(max_tiles)}\n"
        f"\tMax Score: {max(scores)}\n"
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
