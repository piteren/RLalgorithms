"""

 2020 (c) piteren

"""
import numpy as np


class Game:

    def __init__(self, bs=4):
        self.bs = bs
        self.board = np.zeros(bs)
        self.all_states = [np.asarray(([0]*bs + [int(e) for e in list(str(bin(i))[2:])])[-bs:]) for i in range(2**bs)]
        print(f'\nGame initialized for board {self.bs}, all possible states ({len(self.all_states)}):')
        for s in self.all_states: print(Game.state_to_str(s))

    def reset(self):
        self.board = np.zeros(self.bs)

    # returns reward
    def play(self, cell):
        if self.board[cell] == 0:
            self.board[cell] = 1
            return 1
        return -1

    def is_over(self):
        return np.average(self.board) == 1

    @staticmethod
    def state_to_str(state: np.ndarray):
        return str(state.astype(np.int).tolist()).replace(' ','')


if __name__ == "__main__":
    game = Game(bs=4)