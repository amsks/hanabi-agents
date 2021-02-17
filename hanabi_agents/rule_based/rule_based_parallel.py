from .ruleset import Ruleset
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
import numpy as np
import random
import timeit
from functools import partial


class ParallelRulebasedAgent():

  def __init__(self, rules, rules_teammates, compute_diversity):
    self.rules = rules
    self.rules_teammates = rules_teammates
    self.compute_diversity = compute_diversity
    self.totalCalls = 0
    self.histogram = [0 for i in range(len(rules) + 1)]

    self.n_agents = len(rules)
    self.agent_turn = -1

    self.agent_id = random.randint(0, 100)
    self.total_moves = 0

    self.rule_times = []
    for i in range(self.n_agents):
      self.rule_times.append([])
      for k in range(len(rules[i])):
        self.rule_times[i].append([rules[i][k].__name__, 0, 0])
    self.total_time = 0
    self.diversities = np.zeros((len(rules), len(rules)))
    self.num_comparisons = 1

  def next_agent(self):
    self.agent_turn = (self.agent_turn + 1) % self.n_agents

  def compute_action_other(self, observation, agent_index):
    for index, rule in enumerate(self.rules_teammates[agent_index]):
      action = rule(observation)
      if action is not None:
        return action, False
    return Ruleset.legal_random(observation), True

  def compute_action(self, observation, agent_index):
    for index, rule in enumerate(self.rules[agent_index]):
      action = rule(observation)
      if action is not None:
        return action, False
    return Ruleset.legal_random(observation), True

  @staticmethod
  def default_diversity(a0, a1, **kwargs):
    return int(a0 != a1 and not (kwargs['was_random0'] and kwargs['was_random1']))

  def update_diversities(self, observation, diversity_fn=None, **kwargs):
    # offset = observation.current_player_offset
    for i in range(self.n_agents):
      for j in range(self.n_agents):
        a0, was_random0 = self.compute_action(observation, i)
        a1, was_random1 = self.compute_action_other(observation, j)
        if not diversity_fn:
          kwargs = dict(was_random0=was_random0, was_random1=was_random1)
          diversity_fn = self.default_diversity
        self.diversities[i][j] += diversity_fn(a0, a1, **kwargs)
        #self.diversities[i][j] += int(a0 != a1 and not (was_random0 and was_random1))

  def get_move(self, observation):
    if observation.current_player_offset == 0:
      if self.compute_diversity:
        self.num_comparisons += 1
        self.update_diversities(observation)
      self.next_agent()
      # print(self.rules[self.agent_turn])
      for index, rule in enumerate(self.rules[self.agent_turn]):
        start_time = timeit.default_timer()
        action = rule(observation)
        end_time = timeit.default_timer()
        self.rule_times[self.agent_turn][index][1] += (end_time - start_time)
        self.rule_times[self.agent_turn][index][2] += 1
        self.total_time += end_time - start_time
        if action is not None:
          self.totalCalls += 1
          return action

      self.totalCalls += 1
      return Ruleset.legal_random(observation)
    return None

  def explore(self, observations):
    actions = pyhanabi.HanabiMoveVector()
    for observation in observations:
      actions.append(self.get_move(observation))
    return actions

  def exploit(self, observations):
    return self.explore(observations)

  def requires_vectorized_observation(self):
    return False

  def add_experience_first(self, o, st):
    pass

  def add_experience(self, o, a, r, st):
    pass

  def update(self):
    pass
