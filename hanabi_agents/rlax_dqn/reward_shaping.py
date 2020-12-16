# imports
from .params import RewardShapingParams
import numpy as np
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi

class ShapingType:
    NONE=0
    RISKY=1
    DISCARD_LAST_OF_KIND=2

# reward shaping class
class RewardShaper:
    
    def __init__(self,
                 params: RewardShapingParams = RewardShapingParams()):
        
        self.params = params
        self.num_ranks = None
        self.unshaped = (0, ShapingType.NONE)

    def shape(self, observations, moves):
        assert len(observations) == len(moves)

        if self.num_ranks == None:
            for obs in observations:
                self.num_ranks = obs.parent_game.num_ranks

        shaped_rewards = [self._calculate(obs, move)for obs, move in zip(observations, moves)]
        return zip(*shaped_rewards)
                    
    def _calculate(self, observation, move):
                
        if move.move_type == pyhanabi.HanabiMove.Type.kPlay:
            return self._play_shape(observation, move)
        if move.move_type == pyhanabi.HanabiMove.Type.kDiscard:
            return self._discard_shape(observation, move)
        if move.move_type in [pyhanabi.HanabiMove.Type.kRevealColor,
                              pyhanabi.HanabiMove.Type.kRevealRank]:
            return self._hint_shape(observation, move)
        else:
            return self.unshaped
            
    def _discard_shape(self, observation, move):

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
    
    def _hint_shape(self, observation, move):
        return self.unshaped
    
    def _play_shape(self, observation, move):
        
        # the move may be illegal, eg. playing a card that is not available in hand
        try:
            prob = observation.playable_percent()[move.card_index]
        except IndexError:
            return self.unshaped
        
        if prob < self.params.min_play_probability:
            return (self.params.w_play_probability, ShapingType.RISKY)

        return self.unshaped
