import random
from nick_2048 import Nick2048


def test_boardenv_random_direction():
    for _ in range(50):
        assert Nick2048.random_direction() in [0, 1, 2, 3]


def test_boardenv_init():
    game = Nick2048()
    nonzero = [v for v in game.board if v != 0]
    assert len(nonzero) == 2
    for v in nonzero:
        assert v in [2, 4]


def test_set_board():
    board = [1, 2, 3, 4, 5]
    game = Nick2048()
    game.set_board(board)
    assert game.board == [1, 2, 3, 4, 5]
    assert not (game.board is board)


def test_board_env_step_one():
    init_state = [2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    # Note the move will add a random 2 or 4 into the board
    state, reward, done = game.step(game.RIGHT)
    assert game.board[3] == 2
    assert game.board[11] == 2
    nonzeros = [v for v in game.board if v != 0]
    assert len(nonzeros) == 3
    for v in nonzeros:
        assert v in [2, 4]


def test_board_env_step_two():
    init_state = [4, 2, 2, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done = game.step(game.RIGHT)
    assert game.board[3] == 4
    assert game.board[2] == 4
    assert game.board[1] == 4


def test_boardenv_move_logic_four_in_a_row():
    # make sure the behavior is correct when a row is full of same values.
    init_state = [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done = game.step(game.RIGHT)
    assert reward == 8
    assert game.board[2] == 4
    assert game.board[3] == 4
    state, reward, done = game.step(game.RIGHT)
    assert reward == 8
    assert game.board[3] == 8


def test_boardenv_done_logic():
    init_state = [16, 8, 16, 4, 4, 2, 4, 8, 32, 2, 32, 4, 4, 16, 4, 8]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done = game.step(game.RIGHT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done = game.step(game.RIGHT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done = game.step(game.LEFT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done = game.step(game.DOWN)
    assert done
    assert reward == 4


def test_boardenv_move_logic_three_in_a_row():
    # make sure the behavior is correct when 3 elts are same in a row
    init_state = [0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done = game.step(game.DOWN)
    assert state == game.board
    assert reward == 4
    assert game.board[13] == 4
    assert game.board[9] == 2


def test_boardenv_fill_on_move_logic():
    # make sure a new piece is added that is either a 2 or a 4
    init_state = [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done = game.step(game.LEFT)
    assert state == game.board
    assert reward == 4
    assert len([v for v in game.board if v != 0]) == 2
