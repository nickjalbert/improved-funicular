import sys
import statistics
import random
import numpy as np
from nick_2048 import Nick2048
from andy_adapter import Andy2048

TRIALS = 2


# nick's original version that is called twice in a row for the diff games
def do_trials(cls, strategy, check_done_fn=None):
    scores = []
    max_tiles = []
    for i in range(TRIALS):
        game = cls()
        curr_board, score, done = game.get_state()
        while not done:
            assert curr_board == game.board
            move = strategy(curr_board)
            prev_board = curr_board[:]
            curr_board, score, done = game.step(move)
            if check_done_fn is not None:
                done = check_done_fn(prev_board, curr_board, score, done)
        scores.append(score)
        max_tiles.append(max(game.board))
    print(
        f"{strategy.info}:\n"
        f"\tMax Tile: {max(max_tiles)}\n"
        f"\tMax Score: {max(scores)}\n"
        f"\tMean Score: {statistics.mean(scores)}\n"
        f"\tMedian Score: {statistics.median(scores)}\n"
        f"\tStandard Dev: {statistics.stdev(scores)}\n"
        f"\tMin Score: {min(scores)}\n"
    )

# synchronous version for seeing how the two game implementations are different!
def do_trials_compare(nick_cls, andy_cls, strategy, check_done_fn=None):
    scores1 = []
    scores2 = []
    max_tiles1 = []
    max_tiles2 = []
    for i in range(TRIALS):
        print("trail %s" % i)
        game1 = nick_cls()
        curr_board1, score1, done1 = game1.get_state()
        game2 = andy_cls.from_init_state(curr_board1)
        curr_board2, score2, done2 = game2.get_state()
        print("\n%s\n\n%s" % (np.array(curr_board1).reshape(4,4), np.array(curr_board2).reshape(4,4)))
        while not done1:
            assert curr_board1 == game1.board
            assert curr_board2 == game2.board
            move = strategy(curr_board1, nick_cls)
            move2 = strategy(curr_board1, andy_cls)
            prev_board1 = curr_board1[:]
            prev_board2 = curr_board2[:]
            # take a step without adding a random tile and compare
            curr_board1, score1, done1 = game1.step(move, dry_run=True, add_new_random_piece=False)
            curr_board2, score2, done2 = game2.step(move, add_new_random_piece=False)
            print("\nNick:\n%s\nAndy\n%s" % (np.array(curr_board1).reshape(4,4), np.array(curr_board2).reshape(4,4)))
            assert curr_board1 == curr_board2
            # now take a real step in nick's and copy results over to Andy's
            curr_board1, score1, done1 = game1.step(move, add_new_random_piece=True)
            curr_board2 = curr_board1
            game2.andy.state = np.array(curr_board1).reshape(4,4)
            assert score1 == game2.score, (score1, game2.score)
            print("\nNickNEW:\n%s\nAndyNEW\n%s" % (np.array(curr_board1).reshape(4,4), np.array(curr_board2).reshape(4,4)))
            assert curr_board1 == curr_board2, "%s\n\n%s" % (np.array(curr_board1).reshape(4,4), np.array(curr_board2).reshape(4,4))
            assert game1.score == game2.score
            if check_done_fn is not None:
                done1 = check_done_fn(prev_board1, curr_board1, score1, done1)
                done2 = check_done_fn(prev_board2, curr_board2, score2, done2)
                #assert done1 == done2, "%s\n%s\n%s\n%s" % (curr_board1, done1, curr_board2, done2)

        scores1.append(score1)
        scores2.append(score1)
        max_tiles1.append(max(game1.board))
        max_tiles2.append(max(game1.board))
    print(
        f"{strategy.info}:\n"
        f"\tMax Tile Nick: {max(max_tiles1)}\n"
        f"\tMax Tile Andy: {max(max_tiles1)}\n"
        f"\tMax Score Nick: {max(scores1)}\n"
        f"\tMax Score Andy: {max(scores2)}\n"
        f"\tMean Score Nick: {statistics.mean(scores1)}\n"
        f"\tMean Score Andy: {statistics.mean(scores2)}\n"
        f"\tMedian Score Nick: {statistics.median(scores1)}\n"
        f"\tMedian Score Andy: {statistics.median(scores2)}\n"
        f"\tStandard Dev Nick: {statistics.stdev(scores1)}\n"
        f"\tStandard Dev Andy: {statistics.stdev(scores2)}\n"
        f"\tMin Score Nick: {min(scores1)}\n"
        f"\tMin Score Andy: {min(scores2)}\n"
    )

def try_only_go_right(cls1, cls2=None):
    def right_fn(board, cls):
        # this is ok since i changed nick's RIGHT/DOWN/LEFT/UP to be integers 0/1/2/3 like Andy's
        return cls.RIGHT

    def right_done(prev, curr, score, done):
        return done or prev == curr

    right_fn.info = "Strategy only moves right"
    if cls2:
        do_trials_compare(cls1, cls2, right_fn, right_done)
    else:
        do_trials(cls1, right_fn, right_done)


def try_random(cls1, cls2):
    def random_fn(board, cls):
        choices = [cls.UP, cls.RIGHT, cls.DOWN, cls.LEFT]
        return random.choice(choices)

    random_fn.info = "Random strategy"
    do_trials_compare(cls1, cls2, random_fn)


def try_down_left(cls1, cls2):
    def down_left_fn(board, cls):
        game = cls()
        game.set_board(board)
        down_left_actions = [cls.DOWN, cls.LEFT]
        right_up_actions = [cls.RIGHT, cls.UP]
        random.shuffle(down_left_actions)
        random.shuffle(right_up_actions)
        for action in down_left_actions + right_up_actions:
            game.set_board(board[:])
            assert game.board == board
            game.step(action)
            if game.board != board:
                return action
        assert False, "Some action should do something"

    down_left_fn.info = "Down Left strategy"

    do_trials_compare(cls1, cls2, down_left_fn)


def do_stats():
    print(f"\nRunning CONCURRENTLY {TRIALS} trials Nick and Andy impl with test go-right-only strategy\n")
    try_only_go_right(Nick2048, Andy2048)
    print("\nRunning CONCURRENTLY {TRIALS} trials with Nick and Andy impl to test each strategy\n")
    try_random(Nick2048, Andy2048)
    try_down_left(Nick2048, Andy2048)
   # print(f"\nRunning {TRIALS} trials with Andy impl to test each strategy\n")
   # try_only_go_right(Andy2048)
   # try_random(Andy2048)
   # try_down_left(Andy2048)



if __name__ == "__main__":
    do_stats()
