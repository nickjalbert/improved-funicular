import time
from envs.nick_2048 import Nick2048

SEED = 13


def update_search_stack(search_stack, game, action_history):
    for action, _, _ in game.get_valid_actions():
        search_stack.append((game.board, game.score, action, action_history))


def test_action_history(action_history, expected_score):
    test_game = Nick2048(random_seed=SEED)
    for action in action_history:
        test_game.step(action)
    assert test_game.done
    assert test_game.score == expected_score


def dfs_search():
    start = time.time()
    search_stack = []
    game = Nick2048(random_seed=SEED)
    update_search_stack(search_stack, game, [])
    max_score = 0
    max_action_history = 0
    complete_games = 0

    while len(search_stack) > 0:
        board, score, action, action_history = search_stack.pop()
        action_history = action_history[:]
        game.set_board(board)
        game.score = score
        game.step(action)
        action_history.append(action)
        update_search_stack(search_stack, game, action_history)
        if game.done:
            complete_games += 1
            if game.score > max_score:
                max_score = game.score
                max_action_history = action_history[:]
            if complete_games % 1001 == 1000:
                print(
                    f"Random seed: {SEED}\n"
                    f"Max action history: {max_action_history}\n"
                    f"Max Score: {max_score}\n"
                    f"Max moves: {len(max_action_history)}\n"
                    f"Elapsed time: {round(time.time() - start, 2)} sec\n"
                    f"Search stack size: {len(search_stack)}\n"
                    f"Complete Games: {complete_games}\n"
                    f"Complete Games: {complete_games}\n"
                    f"Current Score: {game.score}\n"
                )
                test_action_history(action_history, game.score)
                test_action_history(max_action_history, max_score)


if __name__ == "__main__":
    dfs_search()
