"""

 2021 (c) piteren

"""
from ptools.lipytools.plots import two_dim
from ptools.R4C.qtable import QTable

from QLearning.simple_board_game import SimpleBoardGame


if __name__ == "__main__":

    game = SimpleBoardGame(bs=4)
    qt = QTable(game)
    r_list = qt.build()

    print(qt)
    qt.test()
    two_dim(r_list)