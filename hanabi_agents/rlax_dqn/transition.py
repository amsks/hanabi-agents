"""This file defines a transition.
Transition from state A to state B includes
  -- observation_tm1 - observation at time t - 1
  -- action_tm1      - action at time t - 1
  -- reward_t        - reward at time t for taking action_tm1 while at observation_tm1
  -- observation_t   - observation at time t.
  -- legal_moves_t   - legal moves for observation_t
  -- terminal_t      - whether the transition is terminal.
"""
#from collections import namedtuple

# Transition = namedtuple(
#     "Transition",
#     ["observation_tm1", "action_tm1", "reward_t", "observation_t", "legal_moves_t", "terminal_t"])

def Transition(obs_tm1, action_tm1, reward_t, obs_t, legal_moves_t, terminal_t):
    return {
        "observation_tm1": obs_tm1,
        "action_tm1": action_tm1,
        "reward_t": reward_t,
        "observation_t": obs_t,
        "legal_moves_t": legal_moves_t,
        "terminal_t": terminal_t
    }
