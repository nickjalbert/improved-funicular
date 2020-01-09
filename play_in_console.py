import sys
import readchar
from nick_2048 import Nick2048
from andy_adapter import Andy2048
from strategies.lookahead import get_lookahead_fn


def run_manual_loop(game, lookahead_fn=None):
    game.render_board()
    key_to_moves = {
        "w": game.UP,
        "A": game.UP,
        "d": game.RIGHT,
        "C": game.RIGHT,
        "s": game.DOWN,
        "B": game.DOWN,
        "a": game.LEFT,
        "D": game.LEFT,
    }
    moves_to_str = {
        game.UP: "Up",
        game.RIGHT: "Right",
        game.DOWN: "Down",
        game.LEFT: "Left",
    }
    should_print = True
    while True:
        if should_print:
            if lookahead_fn:
                suggested_action = lookahead_fn(game.board[:])
            print(f"Score: {game.score}")
            print("Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?")
            if lookahead_fn:
                action_str = moves_to_str[suggested_action]
                print(f"Suggested move (spacebar to accept): {action_str}")
            should_print = False
        move = readchar.readchar()
        if move in ["\x03", "\x04", "\x1a"]:
            print("Exiting...")
            break
        if move == "\x20" and lookahead_fn:
            move_items = key_to_moves.items()
            move = [k for k, v in move_items if v == suggested_action][0]
        if move not in key_to_moves:
            continue
        print(f"Move: {moves_to_str[key_to_moves[move]]}")
        should_print = True
        _, _, done = game.step(key_to_moves[move])
        print()
        game.render_board()
        if done:
            print("Game over!")
            break


def play_with_lookahead():
    game = Nick2048()
    lookahead_fn = get_lookahead_fn(Nick2048, 5)
    run_manual_loop(game, lookahead_fn)


def play_nick_version():
    game = Nick2048()
    run_manual_loop(game)


def play_andy_version():
    game = Andy2048()
    run_manual_loop(game)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["andy", "nick", "lookahead"]:
        print(f"\nUsage:")
        print(f"\tpython {sys.argv[0]} [andy | nick | lookahead]")
        print()
        print(f"For example:")
        print(f"\tpython {sys.argv[0]} andy # plays Andy's version")
        print(f"\tpython {sys.argv[0]} nick # plays Nick's version")
        print(
            f"\tpython {sys.argv[0]} lookahead "
            f"# Plays with lookahead to suggest moves"
        )
        print()
        sys.exit(0)
    if sys.argv[1] == "nick":
        play_nick_version()
    elif sys.argv[1] == "andy":
        play_andy_version()
    elif sys.argv[1] == "lookahead":
        play_with_lookahead()
    else:
        raise Exception(f"Unknown version: {sys.argv[1]}")
