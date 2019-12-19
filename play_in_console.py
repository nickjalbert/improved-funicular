import sys
import readchar
from nick_2048 import Nick2048
from andy_adapter import Andy2048


def run_manual_loop(game):
    render_board(game)
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
        render_board(game)
        if done:
            print("Game over!")
            break

def render_board(game):
    def j(el):
        return str(el).rjust(5)

    p = ["·" if x == 0 else x for x in game.board]
    print(
        f"{j(p[0])} {j(p[1])} {j(p[2])} {j(p[3])}\n"
        f"{j(p[4])} {j(p[5])} {j(p[6])} {j(p[7])}\n"
        f"{j(p[8])} {j(p[9])} {j(p[10])} {j(p[11])}\n"
        f"{j(p[12])} {j(p[13])} {j(p[14])} {j(p[15])}\n\n"
    )

def play_nick_version():
    game = Nick2048()
    run_manual_loop(game)


def play_andy_version():
    game = Andy2048()
    run_manual_loop(game)

if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['andy', 'nick']:
        print(f'\nUsage:')
        print(f'\tpython {sys.argv[0]} [andy|nick]')
        print()
        print(f'For example:')
        print(f'\tpython {sys.argv[0]} andy # plays Andy\'s version')
        print(f'\tpython {sys.argv[0]} nick # plays Nick\'s version')
        print()
        sys.exit(0)
    if sys.argv[1] == 'nick':
        play_nick_version()
    elif sys.argv[1] == 'andy':
        play_andy_version()
    else:
        raise Exception(f'Unknown version: {sys.argv[1]}')


