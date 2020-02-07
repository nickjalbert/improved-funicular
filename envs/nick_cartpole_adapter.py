from decimal import Decimal
import gym


class NickCartpoleAdapter:
    STATE_LEN = 4
    action_space = (0, 1)

    def __init__(self):
        self.env = gym.make("CartPole-v0")
        self.max_steps = 200
        self.reset()

    def reset(self):
        self.state = tuple(self.env.reset())
        self.step_count = 0
        self.total_reward = 0
        self.env_is_done = False

    def step(self, action):
        assert action in self.action_space
        next_state, reward, done, _ = self.env.step(action)
        self.state = tuple(next_state)
        self.total_reward += int(reward)
        self.env_is_done = bool(done)
        self.step_count += 1
        return self.state, int(reward), self.done, None

    def set_board(self, state):
        pass

    def get_valid_actions(self):
        return ((0, None, None), (1, None, None))

    def get_state(self):
        return self.state, self.score, self.done

    @classmethod
    def get_canonical_afterstate(cls, state, action):
        return tuple([round(Decimal(el), 2) for el in state])

    @property
    def score(self):
        return self.total_reward

    @property
    def board(self):
        return self.state

    @property
    def done(self):
        return self.step_count > self.max_steps or self.env_is_done
