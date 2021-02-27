"""Simple Agent."""
import random
import numpy as np
from collections import Counter
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi

def playable_card(card, fireworks):
    """A card is playable if it can be placed on the fireworks pile."""

    # if color unknown, check if card rank can be played on each pile
    if (card.color == pyhanabi.HanabiCard.ColorType.kUnknownColor 
        and card.rank != pyhanabi.HanabiCard.RankType.kUnknownRank):
        for fireworks_rank in fireworks:
            if fireworks_rank == card.rank:
                continue
            else:
                return False
        return True

    # if color and rank unknown, card is categorized as not playable
    if (card.color == pyhanabi.HanabiCard.ColorType.kUnknownColor 
        and card.rank == pyhanabi.HanabiCard.RankType.kUnknownRank):
        return False

    # if color known check if rank of card matches the pile
    else:
        return card.rank == fireworks[card.color]


def useless_card(card, fireworks, max_fireworks):
    """ card is useless if it cannot be played on fireworks anymore"""
    
    # if color unknown, check if card rank is useless for each pile
    if (card.color == pyhanabi.HanabiCard.ColorType.kUnknownColor 
        and card.rank != pyhanabi.HanabiCard.RankType.kUnknownRank):
        for fireworks_rank, max_rank in zip(fireworks, max_fireworks):
            if card.rank < fireworks_rank or card.rank >= max_rank: 
                continue
            else:
                return False
        return True

    # if color and rank unknown, card is categorized as not useless
    if (card.color == pyhanabi.HanabiCard.ColorType.kUnknownColor 
        and card.rank == pyhanabi.HanabiCard.RankType.kUnknownRank):
        return False

    # if color known check if rank of card matches the pile
    else:
        return card.rank < fireworks_rank or card.rank >= max_rank 


def get_plausible_cards(observation, player_offset, card_index):
   
    # get knowledge of card defined by player_offset and card__index
    card_knowledge = observation.hands[player_offset].knowledge
    hidden_card = card_knowledge[card_index]
    
    plausible_cards = []
    
    #loop through all color/card combinations and add possible cards to output vecotr
    for color_index in range(observation.parent_game.num_colors):
        for rank_index in range(observation.parent_game.num_ranks):
      
            if (hidden_card.color_plausible(color_index) 
                and hidden_card.rank_plausible(rank_index)):
                
                plausible_card = pyhanabi.HanabiCard(
                    pyhanabi.HanabiCard.ColorType(color_index),
                    pyhanabi.HanabiCard.RankType(rank_index)
                )
                plausible_cards.append(plausible_card)
    return plausible_cards


# Note: max fireworks from 0 to 5, whereas rank goes from 0 to 4
def get_max_fireworks(observation):
    """ if all cards of one type discarded, then max value of fireworks
    pile is reduced """
    
    discarded_cards = Counter(observation.discard_pile)
    num_colors = observation.parent_game.num_colors
    num_ranks = observation.parent_game.num_ranks
    max_fireworks = [num_ranks for i in range(num_colors)]
    num_in_deck_by_rank = [observation.parent_game.number_card_instances(0, i) 
                           for i in range(num_ranks)]
  
    for card, card_counter in discarded_cards.items():
        if card_counter >= num_in_deck_by_rank[card.rank]:
            if max_fireworks[card.color] > card.rank:
                max_fireworks[card.color] = card.rank

    return max_fireworks


class MoveGenerator():
    
    @staticmethod
    def get_discard_move(card_index):
        return pyhanabi.HanabiMove(
              pyhanabi.HanabiMove.Type.kDiscard,
              card_index, # card index
              -1, # target offset
              pyhanabi.HanabiCard.ColorType.kUnknownColor, # color
              pyhanabi.HanabiCard.RankType.kUnknownRank # rank
        )
        
    @staticmethod
    def get_play_move(card_index):
        return pyhanabi.HanabiMove(
              pyhanabi.HanabiMove.Type.kPlay,
              card_index, # card index
              -1, # target offset
              pyhanabi.HanabiCard.ColorType.kUnknownColor, # color
              pyhanabi.HanabiCard.RankType.kUnknownRank # rank
        )
        
    @staticmethod
    def get_reveal_color_move(target_offset, color):
        return pyhanabi.HanabiMove(
              pyhanabi.HanabiMove.Type.kRevealColor,
              -1, # card index
              target_offset, # target offset
              color, # color
              pyhanabi.HanabiCard.RankType.kUnknownRank # rank
        )

    @staticmethod
    def get_reveal_rank_move(target_offset, rank):
        return pyhanabi.HanabiMove(
              pyhanabi.HanabiMove.Type.kRevealRank,
              -1, # card index
              target_offset, # target offset
              pyhanabi.HanabiCard.ColorType.kUnknownColor, # color
              rank # rank
        )

class Ruleset():

    @staticmethod
    def discard_oldest_first(observation):
        
        if observation.information_tokens < observation.parent_game.max_information_tokens:
            return MoveGenerator.get_discard_move(0)
        return None

    #Note: this is not identical to the osawa rule implemented in the Fossgalaxy framework, 
    #as there the rule only takes into account explicitly known colors and ranks
    @staticmethod
    def osawa_discard(observation):
        
        if observation.information_tokens == observation.parent_game.max_information_tokens:
            return None
      
        fireworks = observation.fireworks
        max_fireworks = get_max_fireworks(observation)

        for card_index, card in enumerate(observation.hands[0].knowledge):

            if (card.color_hinted() and not card.rank_hinted()):
                if fireworks[card.color] == observation.parent_game.num_ranks:
                    return MoveGenerator.get_discard_move(card_index)

            if (card.color_hinted() and card.rank_hinted()):
                if (card.rank < fireworks[card.color] or card.rank >= max_fireworks[card.color]):
                    return MoveGenerator.get_discard_move(card_index)

            if card.rank_hinted():
                if card.rank < min(fireworks):
                    return MoveGenerator.get_discard_move(card_index)
                
        for card_index in range(len(observation.hands[0])):
            
            plausible_cards = get_plausible_cards(observation, 0, card_index)
            eventually_playable = False
            for card in plausible_cards:
                if (card.rank < max_fireworks[card.color]):
                    eventually_playable = True
                    break
            if not eventually_playable:
                return MoveGenerator.get_discard_move(card_index)

        return None

    # Note: this rule only looks at the next player on purpose, 
    # for compatibility with the Fossgalaxy implementation. Prioritizes color
    @staticmethod
    def tell_unknown(observation):
        
        if observation.information_tokens == 0:
            return None
        
        PLAYER_OFFSET = 1
        their_hand = observation.hands[PLAYER_OFFSET]

        for card_index, card in enumerate(their_hand.knowledge):
            
            if not card.color_hinted():
                return MoveGenerator.get_reveal_color_move(
                    PLAYER_OFFSET, pyhanabi.HanabiCard.ColorType(card.color))
                
            if not card.rank_hinted():
                return MoveGenerator.get_reveal_rank_move(
                    PLAYER_OFFSET, pyhanabi.HanabiCard.RankType(card.rank))

        return None

    # Note: this rule only looks at the next player on purpose, 
    # for compatibility with the Fossgalaxy implementation. Prioritizes color
    @staticmethod
    def tell_randomly(observation):
        
        if observation.information_tokens == 0:
            return None
        
        PLAYER_OFFSET = 1
        their_hand = observation.hands[PLAYER_OFFSET]
        # choose random card from hand
        card_index = random.randint(0, len(their_hand) - 1)
        card = their_hand.cards[card_index]
        
        # choose random move (reveal color or reveal rank)
        random_move = random.randint(0, 1)
        if random_move == 0: # reveal rank
            return MoveGenerator.get_reveal_rank_move(
                PLAYER_OFFSET, pyhanabi.HanabiCard.RankType(card.rank))
            
        else: # reveal color
            return MoveGenerator.get_reveal_color_move(
                PLAYER_OFFSET, pyhanabi.HanabiCard.ColorType(card.color))
            
        return None

    @staticmethod
    def play_safe_card(observation):
      
        fireworks = observation.fireworks
        hand = observation.hands[0]
    
        for card_index, hint in enumerate(hand.knowledge):
            
            # determine if each plausible card is definitely playable
            plausible_cards = get_plausible_cards(observation, 0, card_index)
            definetly_playable = True
            for plausible in plausible_cards:
                if not playable_card(plausible, fireworks):
                    definetly_playable = False
                    break
                
            if definetly_playable:
                return MoveGenerator.get_play_move(card_index)

        return None

    @staticmethod
    def play_if_certain(observation):
      
        fireworks = observation.fireworks
        for card_index, knowledge in enumerate(observation.hands[0].knowledge):
            
            if knowledge.color_hinted() and knowledge.rank_hinted():
                if  knowledge.rank == fireworks[knowledge.color]:
                    return MoveGenerator.get_play_move(card_index)
        
        return None

    # Prioritizes Rank
    @staticmethod
    def tell_playable_card_outer(observation):
        
        if observation.information_tokens == 0:
            return None
        
        fireworks = observation.fireworks
        # Check if there are any playable cards in the hands of the opponents.
        for player_offset in range(1, observation.parent_game.num_players):
            
            player_hand = observation.hands[player_offset]
            
            # Check if a card in the hand of the opponent is playable and decide on action
            for card_index, (card, hint) in enumerate(zip(player_hand.cards, player_hand.knowledge)):
                card_playable = playable_card(card, fireworks)
                
                # give a rank hint
                if card_playable and not hint.rank_hinted():
                    return MoveGenerator.get_reveal_rank_move(
                        player_offset, pyhanabi.HanabiCard.RankType(card.rank))
                
                # give a color hint
                elif card_playable and not hint.color_hinted():
                    return MoveGenerator.get_reveal_color_move(
                        player_offset, pyhanabi.HanabiCard.ColorType(card.color))

        return None

    @staticmethod
    def tell_dispensable_factory(min_information_tokens=8):
        
        def tell_dispensable(observation):
            
            if observation.information_tokens <= min_information_tokens:
                return None
            elif observation.information_tokens == 0:
                return None
                
            fireworks = observation.fireworks
            
            # Check if there are any playable cards in the hands of the opponents.
            for player_offset in range(1, observation.parent_game.num_players):
                player_hand = observation.hands[player_offset]
            
                # Check if the card in the hand of the opponent is discardable
                for card_index, (card, hint) in enumerate(zip(player_hand.cards, player_hand.knowledge)):
            
                    # all cards of color have been played and color not hinted >> color hint
                    if (not hint.color_hinted() 
                        and fireworks[card.color] == observation.parent_game.num_ranks):
                        return MoveGenerator.get_reveal_color_move(player_offset, card.color)
                    
                    # rank has been played for each color and rank not hinted >> rank hint
                    if not hint.rank_hinted() and card.rank < min(fireworks):
                        return MoveGenerator.get_reveal_rank_move(player_offset, card.rank)
            
                    # rank has been played for card color
                    if card.rank < fireworks[card.color]:
                        # rank is known and color unknown >> hint color
                        if not hint.color_hinted() and hint.rank_hinted():
                            return MoveGenerator.get_reveal_color_move(player_offset, card.color)
                            
                        if hint.color_hinted() and not hint.rank_hinted():
                            return MoveGenerator.get_reveal_rank_move(player_offset, card.rank)
                        
            return None
    
        return tell_dispensable

    # As far as I can tell, this is identical to Tell Playable Outer
    @staticmethod
    def tell_anyone_useful_card(observation):
        return Ruleset.tell_playable_card_outer(observation)

    @staticmethod
    def tell_anyone_useless_card(observation):
        
        if observation.information_tokens == 0:
            return None
        
        fireworks = observation.fireworks
        max_fireworks = get_max_fireworks(observation)
        
        for player_offset in range(1, observation.parent_game.num_players):
            player_hand = observation.hands[player_offset]
        
            for card_index, (card, hint) in enumerate(zip(player_hand.cards, player_hand.knowledge)):
                if useless_card(card ,fireworks, max_fireworks):
                    
                    if not hint.color_hinted():
                        return MoveGenerator.get_reveal_color_move(player_offset, card.color)
                    
                    if not hint.rank_hinted():
                        return MoveGenerator.get_reveal_rank_move(player_offset, card.rank)
        return None

    # Note: this follows the version of the rule that's used on VanDenBergh, 
    # which does not take into account whether or not they already know that information
    @staticmethod
    def tell_most_information(observation):

        if observation.information_tokens == 0:
            return None
        
        best_action = None
        max_affected = -1
        
        for player_offset in range(1, observation.parent_game.num_players):
            
            player_hand = observation.hands[player_offset]
            for card_index, (card, hint) in enumerate(zip(player_hand.cards, player_hand.knowledge)):
          
                affected_colors = 0
                affected_ranks = 0
                
                # count number of cards with same color /rank
                for other_card, other_hint in zip(player_hand.cards, player_hand.knowledge):
                    if card.color == other_card.color and not other_hint.color_hinted():
                        affected_colors += 1
                    if card.rank  == other_card.rank and not other_hint.rank_hinted():
                        affected_ranks+=1
                
                # update best action if more cards are affected than with current best action     
                if affected_colors > max_affected:
                    max_affected = affected_colors
                    best_action =  MoveGenerator.get_reveal_color_move(player_offset, card.color)
                    
                if affected_ranks > max_affected:
                    max_affected = affected_ranks
                    best_action = MoveGenerator.get_reveal_rank_move(player_offset, card.rank)

        return best_action

    #tells rank or color at random if tell_rank is unspecified, tells rank if set to 1, color if set to 0.
    @staticmethod
    def tell_playable_card(observation, tell_rank = random.randint(0,1)):
        
        if observation.information_tokens == 0:
            return None
        
        fireworks = observation.fireworks

        # Check if there are any playable cards in the hands of the opponents.
        for player_offset in range(1, observation.parent_game.num_players):
            player_hand = observation.hands[player_offset]

            for card_index, (card, hint) in enumerate(zip(player_hand.cards, player_hand.knowledge)):
                if playable_card(card, fireworks):
                    
                    # if rank is supposed to be told and rank unknown
                    if tell_rank and not hint.rank_hinted():
                        return MoveGenerator.get_reveal_rank_move(player_offset, card.rank)
                    
                    # if color is supposed to be told or rank already known
                    elif not hint.color_hinted():
                        return MoveGenerator.get_reveal_color_move(player_offset, card.color)
                    
                    # if color is supposed to be told, but already known, hint about rank instead
                    elif not hint.rank_hinted():
                        return MoveGenerator.get_reveal_rank_move(player_offset, card.rank)

        return None

    @staticmethod
    def legal_random(observation):
        """Act based on an observation."""
        return random.choice(observation.legal_moves)

    @staticmethod
    def discard_randomly(observation):
        
        if observation.information_tokens < observation.parent_game.max_information_tokens:
            hand = observation.hands[0]
            hand_size = len(hand)
            discard_index = random.randint(0,hand_size-1)
            return MoveGenerator.get_discard_move(discard_index)
        return None

    @staticmethod
    def play_probably_safe_factory(treshold = 0.95, require_extra_lives = False):
        def play_probably_safe_treshold(observation):
            
            # select card with highest playability
            playability_vector = observation.playable_percent()
            card_index = np.argmax(playability_vector)

            if not require_extra_lives or observation.life_tokens >1:
                if playability_vector[card_index]>=treshold:
                    return MoveGenerator.get_play_move(card_index)
            return None
        
        return play_probably_safe_treshold

    @staticmethod
    def discard_probably_useless_factory(treshold = 0.75):
        def play_probably_useless_treshold(observation):
            
            if observation.information_tokens < observation.parent_game.max_information_tokens:
                # select card that is most probably useless
                probability_useless = observation.discardable_percent()
                card_index = np.argmax(probability_useless)
        
                if probability_useless[card_index]>=treshold:
                    return MoveGenerator.get_discard_move(card_index)
                return None

        return play_probably_useless_treshold

    # "Hail Mary" rule used by agent Piers, play card with highest playability when game cannot be lost
    @staticmethod
    def hail_mary(observation):
        if (observation.deck_size == 0 and observation.life_tokens > 1):
            return Ruleset.play_probably_safe_factory(0.0)(observation)
        return None
