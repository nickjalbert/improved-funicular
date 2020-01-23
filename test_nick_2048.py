from envs.nick_2048 import Nick2048


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
    board = tuple([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    game = Nick2048()
    game.set_board(board)
    assert game.board == board


def test_board_env_step_one():
    init_state = tuple([2, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    # Note the move will add a random 2 or 4 into the board
    state, reward, done, _ = game.step(game.RIGHT)
    assert game.board[3] == 2
    assert game.board[11] == 2
    nonzeros = [v for v in game.board if v != 0]
    assert len(nonzeros) == 3
    for v in nonzeros:
        assert v in [2, 4]


def test_board_env_step_two():
    init_state = tuple([4, 2, 2, 4, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done, _ = game.step(game.RIGHT)
    assert game.board[3] == 4
    assert game.board[2] == 4
    assert game.board[1] == 4


def test_boardenv_move_logic_four_in_a_row():
    # make sure the behavior is correct when a row is full of same values.
    init_state = tuple([2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done, _ = game.step(game.RIGHT)
    assert reward == 8
    assert game.board[2] == 4
    assert game.board[3] == 4
    state, reward, done, _ = game.step(game.RIGHT)
    assert reward == 8
    assert game.board[3] == 8


def test_boardenv_done_logic():
    init_state = tuple([16, 8, 16, 4, 4, 2, 4, 8, 32, 2, 32, 4, 4, 16, 4, 8])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done, _ = game.step(game.RIGHT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done, _ = game.step(game.RIGHT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done, _ = game.step(game.LEFT)
    assert state == game.board
    assert state == init_state
    assert not done
    assert reward == 0
    state, reward, done, _ = game.step(game.DOWN)
    assert done
    assert reward == 4


def test_boardenv_move_logic_three_in_a_row():
    # make sure the behavior is correct when 3 elts are same in a row
    init_state = tuple([0, 2, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done, _ = game.step(game.DOWN)
    assert state == game.board
    assert reward == 4
    assert game.board[13] == 4
    assert game.board[9] == 2


def test_boardenv_fill_on_move_logic():
    # make sure a new piece is added that is either a 2 or a 4
    init_state = tuple([2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state
    state, reward, done, _ = game.step(game.LEFT)
    assert state == game.board
    assert reward == 4
    assert len([v for v in game.board if v != 0]) == 2


def test_set_board_makes_copy():
    init_state = tuple([2, 2, 0, 0, 2, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(init_state)
    assert game.board == init_state


def test_get_valid_actions():
    # UP, DOWN, LEFT, RIGHT is valid
    all_board = tuple([2, 4, 8, 16, 2, 8, 16, 32, 32, 16, 8, 4, 32, 32, 4, 8])
    game = Nick2048()
    game.set_board(all_board)
    all_actions = [(a, r) for (a, r, b) in game.get_valid_actions()]
    assert game.board == all_board
    assert (game.UP, 68) in all_actions
    assert (game.DOWN, 68) in all_actions
    assert (game.RIGHT, 64) in all_actions
    assert (game.LEFT, 64) in all_actions
    for (a, r, b) in Nick2048.get_valid_actions_from_board(all_board):
        assert (a, r) in all_actions
    # No valid actions
    no_board = tuple([2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256])
    game.set_board(no_board)
    no_actions = [(a, r) for (a, r, b) in game.get_valid_actions()]
    assert game.board == no_board
    assert len(no_actions) == 0
    for (a, r, b) in Nick2048.get_valid_actions_from_board(no_board):
        assert (a, r) in no_actions
    # DOWN or RIGHT is valid
    dr_board = tuple([2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    game.set_board(dr_board)
    some_actions = [(a, r) for (a, r, b) in game.get_valid_actions()]
    assert game.board == dr_board
    assert len(some_actions) == 2
    assert (game.DOWN, 0) in some_actions
    assert (game.RIGHT, 0) in some_actions
    for (a, r, b) in Nick2048.get_valid_actions_from_board(dr_board):
        assert (a, r) in some_actions


def test_get_valid_actions_by_reward():
    # UP, DOWN, LEFT, RIGHT is valid
    board = tuple([2, 4, 4, 2, 2, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    game = Nick2048()
    game.set_board(board)
    action_rewards = [(a, r) for (a, r, b) in game.get_valid_actions_by_reward()]
    assert game.board == board
    left_right = [(game.LEFT, 24), (game.RIGHT, 24)]
    up_down = [(game.UP, 4), (game.DOWN, 4)]
    assert action_rewards[0] in left_right
    assert action_rewards[1] in left_right
    assert action_rewards[2] in up_down
    assert action_rewards[3] in up_down
    for (a, r, b) in Nick2048.get_valid_actions_by_reward_from_board(board):
        assert (a, r) in action_rewards


def test_rotate_board():
    # 2 0 4 8
    # 2 0 0 0
    # 4 4 0 0
    # 0 0 0 8
    board = tuple([2, 0, 4, 8, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 8])
    result_90 = Nick2048.rotate_board_right(board)
    # 0 4 2 2
    # 0 4 0 0
    # 0 0 0 4
    # 8 0 0 8
    assert result_90 == tuple([0, 4, 2, 2, 0, 4, 0, 0, 0, 0, 0, 4, 8, 0, 0, 8])
    result_180 = Nick2048.rotate_board_right(result_90)
    # 8 0 0 0
    # 0 0 4 4
    # 0 0 0 2
    # 8 4 0 2
    assert result_180 == tuple([8, 0, 0, 0, 0, 0, 4, 4, 0, 0, 0, 2, 8, 4, 0, 2])
    result_270 = Nick2048.rotate_board_right(result_180)
    # 8 0 0 8
    # 4 0 0 0
    # 0 0 4 0
    # 2 2 4 0
    assert result_270 == tuple([8, 0, 0, 8, 4, 0, 0, 0, 0, 0, 4, 0, 2, 2, 4, 0])
    result_360 = Nick2048.rotate_board_right(result_270)
    assert result_360 == board


def test_reflect_board():
    # 2 0 4 8
    # 2 0 0 0
    # 4 4 0 0
    # 0 0 0 8
    board = tuple([2, 0, 4, 8, 2, 0, 0, 0, 4, 4, 0, 0, 0, 0, 0, 8])
    reflected_y_1 = Nick2048.reflect_board_across_y(board)
    # 8 4 0 2
    # 0 0 0 2
    # 0 0 4 4
    # 8 0 0 0
    assert reflected_y_1 == tuple([8, 4, 0, 2, 0, 0, 0, 2, 0, 0, 4, 4, 8, 0, 0, 0])
    reflected_y_2 = Nick2048.reflect_board_across_y(reflected_y_1)
    assert reflected_y_2 == board
    reflected_x_1 = Nick2048.reflect_board_across_x(board)
    # 0 0 0 8
    # 4 4 0 0
    # 2 0 0 0
    # 2 0 4 8
    assert reflected_x_1 == tuple([0, 0, 0, 8, 4, 4, 0, 0, 2, 0, 0, 0, 2, 0, 4, 8])
    reflected_x_2 = Nick2048.reflect_board_across_x(reflected_x_1)
    assert reflected_x_2 == board
