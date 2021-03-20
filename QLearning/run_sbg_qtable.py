"""

 2021 (c) piteren

"""
from ptools.lipytools.plots import two_dim
from ptools.R4C.qlearning.qtable import QTable

from QLearning.simple_board_game import SimpleBoardGame


if __name__ == "__main__":

    game = SimpleBoardGame(board_size=6)
    qt = QTable(game)
    print(qt)

    r_list = qt.train(batch_size=1)
    print(qt)
    qt.test()
    two_dim(r_list)