"""

 2020 (c) piteren

"""
import numpy as np
import pandas as pd
import random
from matplotlib import pyplot as plt

from QLearning.game_envy import Game


def build_QTable(game):

    num_of_games = 2000
    epsilon = 0.5
    gamma = 0.9

    q_table = pd.DataFrame(0, index=np.arange(len(game.board)), columns=[game.state_to_str(s) for s in game.all_states])
    print(q_table)

    r_list = []  # store the total reward of each game so we can plot it later
    for g in range(num_of_games):
        total_reward = 0
        game.reset()
        while not game.is_over():
            state = np.copy(game.board) # save initial state copy
            if random.random() < epsilon:   action = random.randrange(len(state))
            else:                           action = q_table[game.state_to_str(state)].idxmax()
            reward = game.play(action)
            total_reward += reward

            next_state = game.board.astype(np.int)
            next_state_max_q_value = q_table[game.state_to_str(next_state)].max()
            q_table.loc[action, game.state_to_str(state)] = reward + gamma * next_state_max_q_value

        r_list.append(total_reward)

    print(q_table)
    return q_table, r_list


def test_QTable(game, q_table):

    for st in game.all_states:
        sst = game.state_to_str(st)
        action = q_table[sst].idxmax()
        pred = str([round(v,3) for v in q_table[sst].tolist()])
        print(f'board: {sst}  predicted Q values: {pred:30s}  best action: {action}  correct action? {st[action] == 0}')


def plot_RList(r_list):
    plt.figure(figsize=(14, 7))
    plt.plot(range(len(r_list)), r_list)
    plt.xlabel('Games played')
    plt.ylabel('Reward')
    plt.show()


if __name__ == "__main__":

    seed = 1546847731  # or try a new seed by using: seed = int(time())
    random.seed(seed)
    print(f'\nSeed: {seed}')

    game = Game(bs=4)

    q_table, r_list = build_QTable(game)
    test_QTable(game,q_table)
    plot_RList(r_list)