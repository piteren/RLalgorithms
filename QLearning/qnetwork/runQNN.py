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

from ptools.neuralmess.nemodel import NEModel

from QLearning.qnetwork.qnnModel import qnn_model
from QLearning.game_envy import Game


class QNN:

    def __init__(
            self,
            game,
            mdict :dict,
            seed=   121):

        self.game = game
        self.nn = NEModel(
            fwd_func=   qnn_model,
            mdict=      mdict,
            verb=       1)

        if 'seed' in mdict: seed = mdict['seed']
        random.seed(seed)

    def train(
            self,
            memory_size=    20,
            num_of_games=   2000,
            epsilon=        0.5,
            batch_size=     10):

        memory = ReplayMemory(memory_size)

        r_list = []
        l_list = []  # same as r_list, but for the loss
        counter = 0  # network training trigger
        for g in range(num_of_games):
            self.game.reset()
            total_reward = 0
            while not self.game.is_over():
                counter += 1
                state = np.copy(self.game.board)
                if random.random() < epsilon:
                    action = random.randrange(len(state))
                else:
                    pred = np.squeeze(self.nn.session.run(
                        fetches=    self.nn['output'],
                        feed_dict=  {self.nn['states_PH']: np.expand_dims(self.game.board, axis=0)}))
                    action = np.argmax(pred)
                reward = self.game.play(action)
                total_reward += reward

                memory.append(
                    {'state':       state,
                     'action':      action,
                     'reward':      reward,
                     'next_state':  np.copy(game.board),
                     'game_over':   self.game.is_over()})

                # Network training
                if counter % batch_size == 0:
                    batch = memory.sample(batch_size)
                    q_target = self.nn.session.run(
                        fetches=    self.nn['output'],
                        feed_dict=  {self.nn['states_PH']:  np.array(list(map(lambda x: x['next_state'], batch)))})
                    terminals = np.array(list(map(lambda x: x['game_over'], batch)))
                    for i in range(terminals.size):
                        if terminals[i]: q_target[i] = np.zeros(game.bs) # set Q-value to 0 for terminal states
                    _, loss = self.nn.session.run(
                        fetches=    [self.nn['optimizer'], self.nn['loss']],
                        feed_dict=  {
                            self.nn['states_PH']:       [e['state'] for e in batch],
                            self.nn['rew_PH']:          [e['reward'] for e in batch],
                            self.nn['enum_actions_PH']: np.array(list(enumerate([e['action'] for e in batch]))),
                            self.nn['q_target_PH']:     q_target})
                    l_list.append(loss)

            r_list.append(total_reward)

        print(f'Final cost (avg10): {sum(l_list[-10:])/10:.3f}')


    def test(self):

        for st in self.game.all_states:
            sst = self.game.state_to_str(st)
            pred = np.squeeze(self.nn.session.run(
                fetches=    self.nn['output'],
                feed_dict=  {self.nn['states_PH']: np.expand_dims(st, axis=0)}))
            action = np.argmax(pred)
            pred = str(list(map(lambda x: round(x, 3), pred)))
            print(f'board: {sst}  predicted Q values: {pred:30s}  best action: {action}  correct action? {st[action] == 0}')


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

    game = Game(bs=4)

    mdict = {
        'input_size':           game.bs,
        'hidden_layers_size':   [20],
        'gamma':                0.9,
        'iLR':                  0.003,
        'seed':                 121}

    qnn = QNN(game=game, mdict=mdict)
    qnn.train()
    qnn.test()