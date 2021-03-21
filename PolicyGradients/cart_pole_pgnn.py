"""

 2020 (c) piteren

"""
import gym

from ptools.R4C.policy_gradients.pgnn_model import PGNN
from ptools.R4C.renvy import PolicyGradientsEnvironment


class CartPolePGE(PolicyGradientsEnvironment):

    def __init__(self):
        self.__gym_envy = gym.make('CartPole-v1')
        self.__is_over = False
        self.__gym_envy.reset()
        print(f'\nInitialised gym CartPole-v1 envy')
        print(f' > state width: {self.get_state_width()}')
        print(f' > max steps: {self.__gym_envy._max_episode_steps}')
        print(f' > every step reward: 1, penalty: -100')

    def get_state(self): return self.__gym_envy.state

    def run(self, action: int) -> float:
        next_state, r, game_over, info = self.__gym_envy.step(action)
        self.__is_over = game_over
        if self.__is_over and self.__gym_envy._elapsed_steps < self.__gym_envy._max_episode_steps: r = -100
        return r

    def is_over(self) -> bool: return self.__is_over

    def reset(self):
        self.__is_over = False
        self.__gym_envy.reset()

    def num_actions(self) -> int: return self.__gym_envy.action_space.n

    def encode_state(self, state):
        return self.__gym_envy.state

    def render(self):
        self.__gym_envy.render()

if __name__ == "__main__":

    envy = CartPolePGE()

    mdict = {
        'hidden_layers':    [12],
        'iLR':              0.001,
        'seed':             121}

    pnn = PGNN(
        envy=               envy,
        mdict=              mdict)
    pnn.train()
    pnn.test(exploit=False)