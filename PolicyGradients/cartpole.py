"""

 2020 (c) piteren

"""
import gym
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.stats import zscore
from sklearn.utils import shuffle
import tensorflow as tf


class AgentNN:

    def __init__(
            self,
            state_size,
            num_of_actions,
            hidden_layers,
            learning_rate,
            seed):

        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        tf.reset_default_graph()
        tf.set_random_seed(seed)

        self.states = tf.placeholder(
            shape=  (None, state_size),
            dtype=  tf.float32,
            name=   'input_states')
        self.acc_r = tf.placeholder(
            shape=  None,
            dtype=  tf.float32,
            name=   'accumulated_rewards')
        self.actions = tf.placeholder(
            shape=  None,
            dtype=  tf.int32,
            name=   'actions')

        layer = self.states
        for i in range(len(hidden_layers)):
            layer = tf.layers.dense(
                inputs=             layer,
                units=              hidden_layers[i],
                activation=         tf.nn.relu,
                kernel_initializer= tf.contrib.layers.xavier_initializer(),
                name=               f'hidden_layer_{i+1}')
        self.logits = tf.layers.dense(
            inputs=             layer,
            units=              num_of_actions,
            activation=         None,#tf.nn.tanh,
            kernel_initializer= tf.contrib.layers.xavier_initializer(),
            name=               'logits')
        self.action_prob = tf.nn.softmax(self.logits)
        self.log_policy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.actions)
        self.cost = tf.reduce_mean(self.acc_r * self.log_policy)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def print_stuff(game, s, every=100):
        if game % every == 0 or game == 1: print(s)


    def train(
            self,
            env,
            gamma=              0.99,
            end_game_reward=    -100):

        data = pd.DataFrame(columns=['game','steps','cost'])

        for g in range(1500):
            game = g+1
            game_over = False
            env.reset()
            states = []
            rewards = []
            actions = []
            steps = 0

            self.print_stuff(game, f'starting game {game}')
            while not game_over:
                steps += 1
                current_state = env.state
                probs = self.sess.run(self.action_prob, feed_dict={self.states: np.expand_dims(current_state, axis=0)}).flatten()
                action = np.random.choice(env.action_space.n, p=probs)
                next_state, r, game_over, _ = env.step(action)
                if game_over and steps < env._max_episode_steps: r = end_game_reward

                # save to memory:
                states.append(current_state)
                rewards.append(r)
                actions.append(action)
            self.print_stuff(game, f'game {game} has ended after {steps} steps.')

            discounted_acc_rewards = np.zeros_like(rewards)
            s = 0.0
            for i in reversed(range(len(rewards))):
                s = s * gamma + rewards[i]
                discounted_acc_rewards[i] = s
            discounted_acc_rewards = zscore(discounted_acc_rewards)

            states, discounted_acc_rewards, actions = shuffle(states, discounted_acc_rewards, actions)
            c, _ = self.sess.run(
                fetches=    [self.cost, self.optimizer],
                feed_dict=  {
                    self.states:    states,
                    self.acc_r:     discounted_acc_rewards,
                    self.actions:   actions})

            self.print_stuff(game, f'cost: {c}\n----------')
            data = data.append({'game':game, 'steps':steps, 'cost':c}, ignore_index=True)

        data['steps_moving_average'] = data['steps'].rolling(window=50).mean()
        plt.figure(figsize=(10, 10))
        plt.plot(data['steps_moving_average'])
        plt.xlabel('steps_moving_average')
        plt.show()

        data['cost_moving_average'] = data['cost'].rolling(window=50).mean()
        plt.figure(figsize=(10, 10))
        plt.plot(data['cost_moving_average'])
        plt.xlabel('cost_moving_average')
        plt.show()


if __name__ == "__main__":

    env = gym.make('CartPole-v1')

    ann = AgentNN(
        state_size=     env.observation_space.shape[0],
        num_of_actions= env.action_space.n,
        hidden_layers=  [12],
        learning_rate=  0.001,
        seed=           121)

    ann.train(env)
