"""

 2020 (c) piteren

"""
from ptools.R4C.renvy import REnvy


class SimpleBoardGame(REnvy):

    def __init__(self, bs=4):
        self.__state = None
        self.reset(bs)

        stsL = [bin(n) for n in range(self.num_states())]
        stsL = [s[2:] for s in stsL]
        stsL = ['0'*(4-len(s))+s for s in stsL]
        self.__all_states = {stsL[ix]:ix for ix in range(self.num_states())}
        self.__all_states_rev = {self.__all_states[k]: k for k in self.__all_states}

    def reset(self, bs=None):
        if bs is None: bs = len(self.__state)
        self.__state = '0'*bs # internal representation of state as str 'xxxx', where x is '0' or '1'

    def num_actions(self) -> int:
        return len(self.__state)

    def num_states(self) -> int:
        return self.num_actions()**2

    def get_state(self) -> int:
        return self.__all_states[self.__state]

    def run(self, action: int):
        if self.__state[action] == '0':
            stL = [e for e in self.__state]
            stL[action] = '1'
            self.__state = ''.join(stL)
            return 1
        return -1

    def evaluate(self, state: int, action: int):
        sts = self.__all_states_rev[state]
        if sts[action] == '0': return 1
        return -1

    def is_over(self):
        return '0' not in self.__state


if __name__ == "__main__":

    game = SimpleBoardGame(bs=4)
    game.run(0)
    game.run(1)
    game.run(2)
    game.run(3)
    print(game.is_over())