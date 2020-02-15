import time
import mlflow
from envs.nick_2048 import Nick2048

SEED = 42
DEPTH_LIMIT = 10


def update_search_queue(search_queue, game, action_history):
    for action, _, _ in game.get_valid_actions():
        search_queue.append((game.board, game.score, action, action_history))


def test_action_history(actions, expected_score=None, expected_tile=None):
    assert expected_score or expected_tile
    test_game = Nick2048(random_seed=SEED)
    for action in actions:
        test_game.step(action)
    if expected_score:
        assert test_game.score == expected_score
    if expected_tile:
        assert max(test_game.board) == expected_tile


def bfs_search():
    search_queue = []
    game = Nick2048(random_seed=SEED)
    update_search_queue(search_queue, game, ())
    state_action_pairs = set()
    max_tile = 0
    max_tile_history = ()
    max_score = 0
    max_score_history = ()
    depth_start_time = time.time()

    curr_depth = 1

    while len(search_queue) > 0:
        board, score, action, action_history = search_queue.pop(0)
        if len(action_history) >= curr_depth:
            depth_time = round(time.time() - depth_start_time, 1)
            print(
                f"Depth: {curr_depth}:"
                f"\n\tMax Tile: {max_tile}"
                f"\n\tMax Score: {max_score}"
                f"\n\tTotal State Action Pairs: {len(state_action_pairs)}"
                f"\n\tDepth Time: {depth_time} sec"
            )
            mlflow.log_metric("Max Tile", max_tile, step=curr_depth)
            mlflow.log_metric("Max Score", max_score, step=curr_depth)
            mlflow.log_metric(
                "Total State Action Pairs", len(state_action_pairs), step=curr_depth
            )
            test_action_history(max_tile_history, expected_tile=max_tile)
            test_action_history(max_score_history, expected_score=max_score)
            max_tile = 0
            max_score = 0
            depth_start_time = time.time()
            curr_depth += 1

        game.set_board(board)
        game.score = score
        game.step(action)

        state_action_pairs.add((board, action))
        action_history = (*action_history, action)

        if max(game.board) > max_tile:
            max_tile = max(game.board)
            max_tile_history = action_history
        if game.score > max_score:
            max_score = game.score
            max_score_history = action_history

        update_search_queue(search_queue, game, action_history)

        if len(action_history) > DEPTH_LIMIT:
            break


if __name__ == "__main__":
    print()
    print(f"DFS with Depth Limit {DEPTH_LIMIT} and random seed {SEED}")
    print()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "depth_limit": DEPTH_LIMIT,
                "random_seed": SEED,
                "desc": "DFS of a deterministic 2048",
            }
        )
        bfs_search()
