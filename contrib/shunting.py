import mlflow
import random
import statistics
from envs.nick_2048 import Nick2048


# mlflow.log_metric("Max Tile", max_tile, step=curr_depth)
# mlflow.log_params( { "desc": "DFS of a deterministic 2048", })


class TargetBoard:
    def __init__(self):
        self.targets = {}

    def add(self, board):
        self.targets[sum(board)] = board

    def get_target(self, candidate):
        if sum(candidate) in self.targets:
            return self.targets[sum(candidate)]
        elif (sum(candidate) + 2) in self.targets:
            return self.targets[sum(candidate) + 2]
        else:
            return None


def get_distance(board1, board2):
    squared_difference = sum((a - b) ** 2 for a, b in zip(board1, board2))
    return squared_difference ** 0.5


def test_shunting():
    targets = TargetBoard()
    game = Nick2048()
    # Target trajectory
    while not game.done:
        action = random.choice([a for a, r, b in game.get_valid_actions()])
        canonical = Nick2048.get_canonical_afterstate(game.board, action)
        game.step(action)
        targets.add(canonical)

    target_max = sum(game.board)

    # Random trajectory
    TRIALS = 1000
    random_distances = []
    random_beat_max = 0
    for _ in range(TRIALS):
        game = Nick2048()
        while not game.done:
            action = random.choice([a for a, r, b in game.get_valid_actions()])
            canonical = Nick2048.get_canonical_afterstate(game.board, action)
            target_board = targets.get_target(canonical)
            if target_board is not None:
                distance = get_distance(target_board, canonical)
                random_distances.append(distance)
            game.step(action)
        if sum(game.board) >= target_max:
            random_beat_max += 1

    shunted_distances = []
    shunted_beat_max = 0
    for _ in range(TRIALS):
        game = Nick2048()
        while not game.done:
            actions = [a for a, r, b in game.get_valid_actions()]
            canonicals = [
                (Nick2048.get_canonical_afterstate(game.board, a), a)
                for a in actions
            ]
            target_board = targets.get_target(canonicals[0][0])
            action = random.choice(actions)
            if target_board is not None:
                distances = [
                    (get_distance(target_board, c), a) for c, a in canonicals
                ]
                shunted_distances.append(min(distances)[0])
                action = min(distances)[1]
            game.step(action)
        if sum(game.board) >= target_max:
            shunted_beat_max += 1

    print(
        f"Random vs Shunted board distances to a given target:"
        f"\n\tRandom ({TRIALS} rollouts):"
        f"\n\t\tmean: {statistics.mean(random_distances)}"
        f"\n\t\tmedian: {statistics.median(random_distances)}"
        f"\n\t\tstdev: {statistics.stdev(random_distances)}"
        f"\n\t\tbeat target board: {random_beat_max}"
        f"\n\tShunted ({TRIALS} rollouts):"
        f"\n\t\tmean: {statistics.mean(shunted_distances)}"
        f"\n\t\tmedian: {statistics.median(shunted_distances)}"
        f"\n\t\tstdev: {statistics.stdev(shunted_distances)}"
        f"\n\t\tbeat target board: {shunted_beat_max}"
    )


if __name__ == "__main__":
    assert (
        get_distance((1, 2, 6), (4, 6, 6)) == 5.0
    ), "Distance function is bad"
    with mlflow.start_run():
        test_shunting()
