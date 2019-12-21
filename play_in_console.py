import sys
import readchar
from nick_2048 import Nick2048
from andy_adapter import Andy2048


def run_manual_loop(game):
    game.render_board()
    moves = {
        "w": (game.UP, "Up"),
        "A": (game.UP, "Up"),
        "d": (game.RIGHT, "Right"),
        "C": (game.RIGHT, "Right"),
        "s": (game.DOWN, "Down"),
        "B": (game.DOWN, "Down"),
        "a": (game.LEFT, "Left"),
        "D": (game.LEFT, "Left"),
    }
    should_print = True
    while True:
        if should_print:
            print(f"Score: {game.score}")
            print("Move (w=UP, d=RIGHT, s=DOWN, a=LEFT, or arrows)?")
            should_print = False
        move = readchar.readchar()
        if move in ["\x03", "\x04", "\x1a"]:
            print("Exiting...")
            break
        if move not in moves:
            continue
        print(f"Move: {moves[move][1]}")
        should_print = True
        _, _, done = game.step(moves[move][0])
        print()
        game.render_board()
        if done:
            print("Game over!")
            break


def play_nick_version():
    game = Nick2048()
    run_manual_loop(game)


def play_andy_version():
    game = Andy2048()
    run_manual_loop(game)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ["andy", "nick"]:
        print(f"\nUsage:")
        print(f"\tpython {sys.argv[0]} [andy|nick]")
        print()
        print(f"For example:")
        print(f"\tpython {sys.argv[0]} andy # plays Andy's version")
        print(f"\tpython {sys.argv[0]} nick # plays Nick's version")
        print()
        sys.exit(0)
    if sys.argv[1] == "nick":
        play_nick_version()
    elif sys.argv[1] == "andy":
        play_andy_version()
    else:
        raise Exception(f"Unknown version: {sys.argv[1]}")
