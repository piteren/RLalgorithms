"""

 2020 (c) piteren

"""
from ptools.lipytools.plots import two_dim
from ptools.R4C.qnn.qnn_model import QNN

from QLearning.simple_board_game import SimpleBoardGame


if __name__ == "__main__":

    mdict = {
        'hidden_layers_size':   [20],
        'gamma':                0.9,
        'iLR':                  0.003,
        'seed':                 121}

    game = SimpleBoardGame(board_size=4)

    qnn = QNN(envy=game, mdict=mdict)
    qnn.test()

    r_list = qnn.train(num_of_games=5000)
    print(f'Final cost (avg10): {sum(qnn.l_list[-10:]) / 10:.3f}')
    two_dim(r_list)
    two_dim(qnn.l_list)
    qnn.test()