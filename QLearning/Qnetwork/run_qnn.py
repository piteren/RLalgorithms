"""

 2020 (c) piteren

"""

from collections import deque
import random
import numpy as np
import os
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ptools.R4C.renvy import REnvy
from ptools.neuralmess.nemodel import NEModel
from ptools.lipytools.plots import two_dim

from QLearning.Qnetwork.qnn_model import qnn_model
from QLearning.simple_board_game import SimpleBoardGame


class QNN:

    def __init__(
            self,
            envy: REnvy,
            mdict :dict,
            seed=   121):

        self.__envy = envy
        self.nn = NEModel(
            fwd_func=   qnn_model,
            mdict=      mdict,
            save_TFD=   '_models',
            verb=       1)

        if 'seed' in mdict: seed = mdict['seed']
        random.seed(seed)

    def train(
            self,
            memory_size=    20,
            num_of_games=   5000,
            epsilon=        0.5,
            batch_size=     10):

        memory = ReplayMemory(memory_size)

        r_list = []
        l_list = []  # same as r_list, but for the loss
        counter = 0  # network training trigger
        for g in range(num_of_games):
            self.__envy.reset()
            total_reward = 0
            while not self.__envy.is_over():
                counter += 1
                state = self.__envy.get_state()
                if random.random() < epsilon:
                    action = random.randrange(self.__envy.num_actions())
                else:
                    feed = {self.nn['state_PH']: [state]}
                    res = self.nn.session.run(
                        fetches=    self.nn['output'],
                        feed_dict=  feed)
                    pred = res[0]
                    action = np.argmax(pred)
                reward = self.__envy.run(action)
                total_reward += reward

                memory.append({
                    'state':        state,
                    'action':       action,
                    'reward':       reward,
                    'next_state':   self.__envy.get_state(),
                    'game_over':    self.__envy.is_over()})

                # Network training
                if counter % batch_size == 0:
                    batch = memory.sample(batch_size)
                    q_target = self.nn.session.run(
                        fetches=    self.nn['output'],
                        feed_dict=  {self.nn['state_PH']: np.array(list(map(lambda x: x['next_state'], batch)))})
                    terminals = np.array(list(map(lambda x: x['game_over'], batch)))
                    for i in range(terminals.size):
                        if terminals[i]: q_target[i] = np.zeros(self.__envy.num_actions()) # set Q-value to 0 for terminal states
                    _, loss = self.nn.session.run(
                        fetches=    [self.nn['optimizer'], self.nn['loss']],
                        feed_dict=  {
                            self.nn['state_PH']:        [e['state'] for e in batch],
                            self.nn['rew_PH']:          [e['reward'] for e in batch],
                            self.nn['enum_actions_PH']: np.array(list(enumerate([e['action'] for e in batch]))),
                            self.nn['q_target_PH']:     q_target})
                    l_list.append(loss)
                    #print(loss)

            r_list.append(total_reward)

        print(f'Final cost (avg10): {sum(l_list[-10:])/10:.3f}')
        two_dim(r_list)
        two_dim(l_list)


    def test(self):

        for st in range(self.__envy.num_states()):
            pred = self.nn.session.run(
                fetches=    self.nn['output'],
                feed_dict=  {self.nn['state_PH']: [st]})
            pred = pred[0]
            action = int(np.argmax(pred))
            pred = str(list(map(lambda x: round(x, 3), pred)))
            print(f'state: {st}  QVs: {pred:30s}  action: {action}  (eval:{self.__envy.evaluate(st, action)})')


class ReplayMemory:

    def __init__(self, size):
        self.memory = deque(maxlen=size)
        self.counter = 0

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        return random.sample(self.memory, n)


if __name__ == "__main__":

    game = SimpleBoardGame(bs=4)

    mdict = {
        'input_size':           game.num_actions(),
        'hidden_layers_size':   [20],
        'gamma':                0.9,
        'iLR':                  0.003,
        'seed':                 121}

    qnn = QNN(envy=game, mdict=mdict)
    qnn.test()
    qnn.train()
    qnn.test()