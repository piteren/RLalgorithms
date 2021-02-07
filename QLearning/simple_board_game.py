"""

 2020 (c) piteren

"""
from typing import Hashable

from QLearning.game import QGame


class SimpleBoardGame(QGame):

    def __init__(self, bs=4):
        self.__state = None
        self.reset(bs)

    def num_actions(self) -> int:
        return len(self.__state)

    def get_state(self) -> Hashable:
        return ''.join(self.__state)

    def play(self, action: int):
        if self.__state[action] == '0':
            self.__state[action] = '1'
            return 1
        return -1

    def evaluate(self, state, action: int):
        if state[action] == '0': return 1
        return -1

    def is_over(self):
        return '0' not in self.__state

    def reset(self, bs=None):
        if bs is None: bs = len(self.__state)
        self.__state = ['0']*bs


if __name__ == "__main__":

    game = SimpleBoardGame(bs=4)
    game.play(0)
    game.play(1)
    game.play(2)
    game.play(3)
    print(game.is_over())