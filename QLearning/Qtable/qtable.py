"""

 2020 (c) piteren

"""
import numpy as np
import random
from typing import Hashable, List

from QLearning.game import QGame


class QTable:

    def __init__(
            self,
            game: QGame,
            seed=   123):

        random.seed(seed)
        self.__game = game
        self.__tbl = None
        self.reset()

    def reset(self):
        self.__tbl = {}  # {state: np.array(QValues)}

    def __init_state(self, state: Hashable):
        if state not in self.__tbl:
            #self.__tbl[state] = np.zeros(self.num_actions, dtype=np.float) # init with 0
            self.__tbl[state] = np.random.random(self.__game.num_actions()) # init with random

    def __upd_QV(self, state: Hashable, action: int, qv):
        if state not in self.__tbl: self.__init_state(state)
        self.__tbl[state][action] = qv

    def __get_QVs(self, state: Hashable) -> np.ndarray:
        if state not in self.__tbl: self.__init_state(state)
        return self.__tbl[state]

    def get_states(self) -> List[Hashable]:
        return sorted(list(self.__tbl.keys()))

    def build(
            self,
            num_of_games=   2000,
            epsilon=        0.5,
            gamma=          0.9):

        r_list = []  # store the total reward of each game so we can plot it later
        for g in range(num_of_games):
            total_reward = 0
            self.__game.reset()
            while not self.__game.is_over():
                state = self.__game.get_state() # save initial state copy
                qv = self.__get_QVs(state)
                if random.random() < epsilon:   action = random.randrange(self.__game.num_actions())
                else:                           action = np.argmax(qv)
                reward = self.__game.play(action)
                total_reward += reward

                next_state = self.__game.get_state()
                next_state_max_q_value = max(self.__get_QVs(next_state))
                new_qv = reward + gamma * next_state_max_q_value
                self.__upd_QV(state, action, new_qv)

            r_list.append(total_reward)

        return r_list

    def test(self):
        for st in self.get_states():
            qv = self.__get_QVs(st)
            action = int(np.argmax(qv))
            pred = str([round(v, 3) for v in qv])
            print(f'state: {st}  QVs: {pred:30s}  action: {action}  (eval:{self.__game.evaluate(st,action)})')

    def __str__(self):
        s = 'QTable:\n'
        for st in self.get_states():
            s += f'{st} : {self.__tbl[st]}\n'
        return s