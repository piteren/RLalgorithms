import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import deque
from time import time


def state_to_str(state):
    return str(state.astype(np.int).tolist()).replace(' ','')


class Game:

    def __init__(self, bs=4):
        self.bs = bs
        self.board = np.zeros(bs)
        self.all_states = [np.asarray(([0]*bs + [int(e) for e in list(str(bin(i))[2:])])[-bs:]) for i in range(2**bs)]
        print(f'\nGame initialized for board {self.bs}, all possible states ({len(self.all_states)}):')
        for s in self.all_states: print(state_to_str(s))

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


def build_QTable(game):

    num_of_games = 2000
    epsilon = 0.5
    gamma = 0.9

    q_table = pd.DataFrame(0, index=np.arange(len(game.board)), columns=[state_to_str(s) for s in game.all_states])
    print(q_table)

    r_list = []  # store the total reward of each game so we can plot it later
    for g in range(num_of_games):
        total_reward = 0
        game.reset()
        while not game.is_over():
            state = np.copy(game.board) # save initial state copy
            if random.random() < epsilon:   action = random.randrange(len(state))
            else:                           action = q_table[state_to_str(state)].idxmax()
            reward = game.play(action)
            total_reward += reward

            next_state = game.board.astype(np.int)
            next_state_max_q_value = q_table[state_to_str(next_state)].max()
            q_table.loc[action, state_to_str(state)] = reward + gamma * next_state_max_q_value

        r_list.append(total_reward)

    print(q_table)
    return q_table, r_list


def test_QTable(game, q_table):

    for st in game.all_states:
        sst = state_to_str(st)
        action = q_table[sst].idxmax()
        pred = str([round(v,3) for v in q_table[sst].tolist()])
        print(f'board: {sst}  predicted Q values: {pred:30s}  best action: {action}  correct action? {st[action] == 0}')


def plot_RList(r_list):
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(r_list)), r_list)
    plt.xlabel('Games played')
    plt.ylabel('Reward')
    plt.show()


class QNetwork:

    def __init__(
            self,
            hidden_layers_size,
            gamma,
            learning_rate,
            input_size=     4,
            output_size=    4):

        self.q_target = tf.placeholder(
            shape=  (None,output_size),
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
            units=              output_size,
            activation=         None,
            kernel_initializer= tf.contrib.layers.xavier_initializer(seed=seed))
        self.predictions = tf.gather_nd(self.output, indices=self.enum_actions)
        self.labels = self.r + gamma * tf.reduce_max(self.q_target, axis=1)
        self.cost = tf.reduce_mean(tf.losses.mean_squared_error(
            labels=         self.labels,
            predictions=    self.predictions))
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(self.cost)


class ReplayMemory:

    def __init__(self, size):
        self.memory = deque(maxlen=size)
        self.counter = 0

    def append(self, element):
        self.memory.append(element)
        self.counter += 1

    def sample(self, n):
        return random.sample(self.memory, n)


def train_QNetwork():
    num_of_games = 2000
    epsilon = 0.5
    gamma = 0.9
    learning_rate = 0.003
    batch_size = 10
    memory_size = 20

    tf.reset_default_graph()
    tf.set_random_seed(seed)
    qnn = QNetwork(hidden_layers_size=[20], gamma=gamma, learning_rate=learning_rate)
    memory = ReplayMemory(memory_size)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

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
                pred = np.squeeze(sess.run(qnn.output, feed_dict={qnn.states: np.expand_dims(game.board, axis=0)}))
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
                q_target = sess.run(
                    qnn.output,
                    feed_dict={qnn.states: np.array(list(map(lambda x: x['next_state'], batch)))})
                terminals = np.array(list(map(lambda x: x['game_over'], batch)))
                for i in range(terminals.size):
                    if terminals[i]: q_target[i] = np.zeros(game.bs) # set Q-value to 0 for terminal states
                _, cost = sess.run(
                    [qnn.optimizer, qnn.cost],
                    feed_dict={
                        qnn.states:       [e['state'] for e in batch],
                        qnn.r:            [e['reward'] for e in batch],
                        qnn.enum_actions: np.array(list(enumerate([e['action'] for e in batch]))),
                        qnn.q_target:     q_target})
                c_list.append(cost)

        r_list.append(total_reward)

    print(f'Final cost (avg10): {sum(c_list[-10:])/10:.3f}')
    return sess, qnn

def test_QNetwork(game, sess, qnn):

    for st in game.all_states:
        sst = state_to_str(st)
        pred = np.squeeze(sess.run(qnn.output, feed_dict={qnn.states: np.expand_dims(st, axis=0)}))
        action = np.argmax(pred)
        pred = str(list(map(lambda x: round(x, 3), pred)))
        print(f'board: {sst}  predicted Q values: {pred:30s}  best action: {action}  correct action? {st[action]==0}')



if __name__ == "__main__":

    seed = 1546847731  # or try a new seed by using: seed = int(time())
    random.seed(seed)
    print(f'\nSeed: {seed}')

    game = Game(bs=4)

    #q_table, r_list = build_QTable(game)
    #test_QTable(game,q_table)
    #plot_RList(r_list)

    sess, qnn = train_QNetwork()
    test_QNetwork(game, sess, qnn)