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

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from ptools.neuralmess.nemodel import NEModel

from PolicyGradients.policy_model import policy_model


class PolicyNN:

    def __init__(self,
            env,
            mdict :dict):

        self.env = env
        self.nn = NEModel(
            fwd_func=   policy_model,
            mdict=      mdict,
            save_TFD=   '_models',
            verb=       1)

    @staticmethod
    def print_stuff(game, s, every=100):
        if game % every == 0 or game == 1: print(s)


    def train(
            self,
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
                probs = self.nn.session.run(
                    fetches=    self.nn['action_prob'],
                    feed_dict=  {self.nn['states_PH']: np.expand_dims(current_state, axis=0)}).flatten()
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
            c, _ = self.nn.session.run(
                fetches=    [self.nn['loss'], self.nn['optimizer']],
                feed_dict=  {
                    self.nn['states_PH']:   states,
                    self.nn['acc_rew_PH']:  discounted_acc_rewards,
                    self.nn['actions_PH']:  actions})

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

    mdict = {
        'state_size':       env.observation_space.shape[0],
        'num_of_actions':   env.action_space.n,
        'hidden_layers':    [12],
        'iLR':              0.001,
        'seed':             121}

    pnn = PolicyNN(env=env, mdict=mdict)
    pnn.train()