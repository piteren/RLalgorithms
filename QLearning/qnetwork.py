"""

 2020 (c) piteren

"""

from collections import deque
import random
import numpy as np
import os
import tensorflow as tf

from QLearning.game_envy import Game


class QNetwork:

    def __init__(
            self,
            input_size,
            hidden_layers_size,
            gamma,
            learning_rate,
            seed):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        random.seed(seed)
        tf.reset_default_graph()
        tf.set_random_seed(seed)

        self.q_target = tf.placeholder(
            shape=  (None,input_size),
            dtype=  tf.float32)
        self.r = tf.placeholder(
            shape=  None,
            dtype=  tf.float32)
        self.states = tf.placeholder(
            shape=  (None,input_size),
            dtype=  tf.float32)
        self.enum_actions = tf.placeholder(
            shape=  (None,2),
            dtype=  tf.int32)

        layer = self.states
        for l in hidden_layers_size:
            layer = tf.layers.dense(
                inputs=             layer,
                units=              l,
                activation=         tf.nn.relu,
                kernel_initializer= tf.contrib.layers.xavier_initializer(seed=seed))
        self.output = tf.layers.dense(
            inputs=             layer,
            units=              input_size,
            activation=         None,
            kernel_initializer= tf.contrib.layers.xavier_initializer(seed=seed))
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(self.q_target, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=         self.labels,
            predictions=    self.predictions))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())



    def train(
            self,
            game,
            memory_size=    20,
            num_of_games=   2000,
            epsilon=        0.5,
            batch_size=     10):

        memory = ReplayMemory(memory_size)

        r_list = []
        c_list = []  # same as r_list, but for the cost
        counter = 0  # network training trigger
        for g in range(num_of_games):
            game.reset()
            total_reward = 0
            while not game.is_over():
                counter += 1
                state = np.copy(game.board)
                if random.random() < epsilon:
                    action = random.randrange(len(state))
                else:
                    pred = np.squeeze(self.sess.run(self.output, feed_dict={self.states: np.expand_dims(game.board, axis=0)}))
                    action = np.argmax(pred)
                reward = game.play(action)
                total_reward += reward

                memory.append(
                    {'state':       state,
                     'action':      action,
                     'reward':      reward,
                     'next_state':  np.copy(game.board),
                     'game_over':   game.is_over()})

                # Network training
                if counter % batch_size == 0:
                    batch = memory.sample(batch_size)
                    q_target = self.sess.run(
                        self.output,
                        feed_dict={self.states: np.array(list(map(lambda x: x['next_state'], batch)))})
                    terminals = np.array(list(map(lambda x: x['game_over'], batch)))
                    for i in range(terminals.size):
                        if terminals[i]: q_target[i] = np.zeros(game.bs) # set Q-value to 0 for terminal states
                    _, cost = self.sess.run(
                        [self.optimizer, self.cost],
                        feed_dict={
                            self.states:       [e['state'] for e in batch],
                            self.r:            [e['reward'] for e in batch],
                            self.enum_actions: np.array(list(enumerate([e['action'] for e in batch]))),
                            self.q_target:     q_target})
                    c_list.append(cost)

            r_list.append(total_reward)

        print(f'Final cost (avg10): {sum(c_list[-10:])/10:.3f}')


    def test(
            self,
            game):

        for st in game.all_states:
            sst = game.state_to_str(st)
            pred = np.squeeze(self.sess.run(self.output, feed_dict={self.states: np.expand_dims(st, axis=0)}))
            action = np.argmax(pred)
            pred = str(list(map(lambda x: round(x, 3), pred)))
            print(
                f'board: {sst}  predicted Q values: {pred:30s}  best action: {action}  correct action? {st[action] == 0}')


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

    qnn = QNetwork(
        input_size=         game.bs,
        hidden_layers_size= [20],
        gamma=              0.9,
        learning_rate=      0.003,
        seed=               121)

    qnn.train(game)
    qnn.test(game)