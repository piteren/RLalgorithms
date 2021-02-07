"""

 2020 (c) piteren

"""
from abc import abstractmethod, ABC
from typing import Hashable


# interface of game that may be managed by QTable
class QGame(ABC):

    # returns number of game actions
    @abstractmethod
    def num_actions(self) -> int: pass

    # returns current state
    @abstractmethod
    def get_state(self) -> Hashable: pass

    # plays action, returns reward
    @abstractmethod
    def play(self, action: int) -> float: pass

    # returns evaluation score of state and action
    @abstractmethod
    def evaluate(self, state, action: int): pass

    # checks if game(episode) is over
    @abstractmethod
    def is_over(self) -> bool: pass

    # resets game
    @abstractmethod
    def reset(self): pass