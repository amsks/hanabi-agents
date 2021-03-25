# imports
from .params import RewardShapingParams
import numpy as np
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from collections import Counter

# deprecated, use reward shaper in hanabi learning environment
# discard shaping contains bugs!

class ShapingType:
    NONE=0
    RISKY=1
    DISCARD_LAST_OF_KIND=2
    CONSERVATIVE=3
    
class RewardShapingParams(NamedTuple):
    
    # conservative agent
    shaper: bool = True
    min_play_probability: float = 0.8
    w_play_penalty: Union[Callable[[int], float], float] = 0
    m_play_penalty: float = 0
    w_play_reward: Union[Callable[[int], float], float] = 0
    m_play_reward: float = 0
    penalty_last_of_kind: float = 0

# reward shaping class
class RewardShaper:
    
    def __init__(self,
                 params: RewardShapingParams = RewardShapingParams()):
        
        if not callable(params.w_play_penalty):
            penalty = params.w_play_penalty
            params = params._replace(w_play_penalty=lambda ts: penalty)
        if not callable(params.w_play_reward):
            reward = params.w_play_reward
            params = params._replace(w_play_reward=lambda ts: reward)
        
        self.params = params
        self.unshaped = (0, ShapingType.NONE)
        
        # auxiliary variables
        self.num_ranks = None
        self._performance = 0
        self._m_play_penalty = 0
        self._m_play_reward = 0
    
    @property
    def performance(self):
        return self._performance
    
    @performance.setter
    def performance(self, performance):
        self._performance = performance
        self._m_play_penalty = self.params.m_play_penalty * self._performance
        self._m_play_reward = self.params.m_play_reward * self._performance

    def shape(self, observations, moves, step):
        
        assert len(observations) == len(moves)

        if self.num_ranks == None:
            for obs in observations:
                self.num_ranks = obs.parent_game.num_ranks
                
        shaped_rewards = [self._calculate(obs, move, step)
                          for obs, move in zip(observations, moves)]
        return zip(*shaped_rewards)
                    
    def _calculate(self, observation, move, step):
                
        if move.move_type == pyhanabi.HanabiMove.Type.kPlay:
            return self._play_shape(observation, move, step)
        if move.move_type == pyhanabi.HanabiMove.Type.kDiscard:
            return self._discard_shape(observation, move, step)
        if move.move_type in [pyhanabi.HanabiMove.Type.kRevealColor,
                              pyhanabi.HanabiMove.Type.kRevealRank]:
            return self._hint_shape(observation, move, step)
        else:
            return self.unshaped
            
    def _discard_shape(self, observation, move, step):
        
        if self.params.penalty_last_of_kind==0:
            return self.unshaped

        discard_pile = observation.discard_pile
        card_index = move.card_index
        discarded_card = observation.card_to_discard(card_index)

        if discarded_card.rank == self.num_ranks -1:
            return (self.params.penalty_last_of_kind, ShapingType.DISCARD_LAST_OF_KIND)
        
        elif len(discard_pile) == 0:
            return self.unshaped
        
        elif discarded_card.rank > 0:
            for elem in discard_pile:
                if discarded_card.rank == elem.rank & discarded_card.color == elem.color:
                    return (self.params.penalty_last_of_kind, ShapingType.DISCARD_LAST_OF_KIND)
            return self.unshaped
        
        else:
            counter = 0
            for elem in discard_pile:
                if elem.rank == 0 & elem.color == discarded_card.color:
                    counter += 1
            if counter == 2:
                return (self.params.penalty_last_of_kind, ShapingType.DISCARD_LAST_OF_KIND)
            else:
                return self.unshaped

    def _hint_shape(self, observation, move, step):
        return self.unshaped
    
    def _play_shape(self, observation, move, step):
        
        try:
            prob = observation.playable_percent()[move.card_index]
        except IndexError:
            return self.unshaped
        
        if prob < self.params.min_play_probability:

            penalty = self.params.w_play_penalty(step) + self._m_play_penalty
            return (penalty, ShapingType.RISKY)

        reward = self.params.w_play_reward(step) + self._m_play_reward
        return (reward, ShapingType.CONSERVATIVE)
    
    def __repr__(self):
        return f"<rlax_dqn.RewardShaper(params={self.params})>"

