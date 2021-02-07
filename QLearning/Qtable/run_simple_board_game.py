"""

 2020 (c) piteren

"""
from ptools.lipytools.plots import two_dim

from QLearning.simple_board_game import SimpleBoardGame
from QLearning.Qtable.qtable import QTable


if __name__ == "__main__":

    game = SimpleBoardGame(bs=4)
    qt = QTable(game)
    r_list = qt.build()

    print(qt)
    qt.test()
    two_dim(r_list)