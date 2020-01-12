import gym
import random

class Base2048(gym.Env):
    @classmethod
    def get_valid_actions_from_board(cls, board):
        test_game = cls()
        test_game.set_board(board)
        return test_game.get_valid_actions()

    @classmethod
    def get_valid_actions_by_reward_from_board(cls, board):
        test_game = cls()
        test_game.set_board(board)
        return test_game.get_valid_actions_by_reward()

    def get_valid_actions(self):
        """Returns list of 3-tuples: [(action, reward, board),...]"""
        test_game = self.__class__()
        valid_actions = []
        all_actions = [self.UP, self.DOWN, self.LEFT, self.RIGHT]
        random.shuffle(all_actions)
        for action in all_actions:
            test_game.set_board(self.board)
            board, reward, _, _= test_game.step(action)
            if board != self.board:
                valid_actions.append((action, reward, board))
        return valid_actions

    def get_valid_actions_by_reward(self):
        return sorted(self.get_valid_actions(), key=lambda x: -1 * x[1])

    def render_board(self):
        def j(el):
            return str(el).rjust(5)

        p = ["·" if x == 0 else x for x in self.board]
        print(
            f"{j(p[0])} {j(p[1])} {j(p[2])} {j(p[3])}\n"
            f"{j(p[4])} {j(p[5])} {j(p[6])} {j(p[7])}\n"
            f"{j(p[8])} {j(p[9])} {j(p[10])} {j(p[11])}\n"
            f"{j(p[12])} {j(p[13])} {j(p[14])} {j(p[15])}\n\n"
        )
