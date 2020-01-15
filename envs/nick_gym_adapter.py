""" A wrapper around Nick2048 that makes it use numpy arrays instead of tuples for board states."""

from envs.nick_2048 import Nick2048
import numpy as np


class Nick2048Gym(Nick2048):
    def reset(self):
        new_board = super().reset()
        return np.asarray(new_board)

    def get_state(self):
        board, score, done = super().get_state()
        return np.asarray(board), score, done

    def step(self, direction):
        obs, reward, done, c = super().step(direction)
        return np.asarray(obs), reward, done, c

    def set_board(self, board):
        if not isinstance(board, tuple):
            board = tuple(board)
        super().set_board(board)
