class Base2048:
    def get_valid_actions(self):
        """Returns list of 2-tuples: [(action, reward),...]"""
        test_game = self.__class__()
        valid_actions = []
        for action in [self.UP, self.DOWN, self.LEFT, self.RIGHT]:
            test_game.set_board(self.board)
            board, reward, _ = test_game.step(action)
            if board != self.board:
                valid_actions.append((action, reward))
        return valid_actions

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
