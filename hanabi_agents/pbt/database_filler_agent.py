from typing import Callable
import numpy as np
import math
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper
from hanabi_agents.rlax_dqn import VectorizedObservationStacker
import gin
from dm_env import specs as dm_specs
from typing import Tuple, Dict, List, Union
import random
import os
from dm_env import TimeStep, StepType
from sqlalchemy import create_engine
from sqlalchemy import Table, Column, MetaData, ForeignKey

import time
import pickle
import sqlite3

"""
### Work in Progress###

An agent which maintains/manages and evaluates a population of DQN/Rainbow agents.

"""

Actions = np.ndarray
Observations = np.ndarray
Rewards = np.ndarray
LegalMoves = np.ndarray


def observation_spec_vec_batch(env_obs_size, pop_size) -> Tuple[dm_specs.BoundedArray, dm_specs.BoundedArray]:
    """Overwrite existing object due to the fact, that the one, that is being passed over from the parallel_environment
        is not editable and the first element of shape is by factor 'population_size' too big.
    """
    return (dm_specs.BoundedArray(shape=(int(env_obs_size.shape[0] / pop_size),
                                         env_obs_size.shape[1]),
                                  dtype=np.int8,
                                  name="agent_observations",
                                  minimum=0, maximum=1),
            dm_specs.BoundedArray(shape=(int(env_obs_size.shape[0] / pop_size),
                                         env_obs_size.shape[1]),
                                  dtype=np.int8,
                                  name="legal_moves",
                                  minimum=0, maximum=1))


# print agent's current parameterset
def agents_status(agents):
    for i, agent in enumerate(agents):
        print('Agent_{} has learning_rate set to {} & buffersize set to {}'.format(i, agent.learning_rate,
                                                                                   agent.buffersize))


class DataGenerationAgent:

    def __init__(
            self,
            n_states,
            env_obs_size,
            env_act_size,
            agent_params: RlaxRainbowParams = RlaxRainbowParams()):

        self.agent_params = agent_params
        self.n_states = n_states
        self.env_act_size = env_act_size
        self.env_obs_size = env_obs_size


        self.states_done = np.full((self.n_states,), False)
        self.states_reset = np.full((self.n_states,), False)
        self.evaluations = np.zeros((self.n_states,))

        self.conn = sqlite3.connect('/mnt/raid/ni/hanabi/obs.db')
        self.c = self.conn.cursor()
        self.c.execute('''CREATE TABLE IF NOT EXISTS obs (obs_obj BLOB, obs_vec BLOB, obs_act_vec BLOB)''')
        # self.c.execute('INSERT INTO obs VALUES (?, ?)', (1, 0))
        self.conn.commit()
        # engine = sqlalchemy.create_engine('sqlite://obs.db')

        '''
        1. If available: Load agents from checkpoints
        '''
       
        # TODO: --later-- start with weights -option
        # if start_with_weights is not None:
        #     print(start_with_weights)
        #     for aid, agent in enumerate(agents):
        #         if "agent_" + str(aid) in start_with_weights:
        #             agent.restore_weights(*(start_with_weights["agent_" + str(aid)]))

        self.agent = DQNAgent(self.env_obs_size,
                                self.env_act_size,
                                agent_params)
        # agent = DQNAgent(self.env_obs_size,
        #                  self.env_act_size,
        #                  custom_params)

    def write_to_db(self):
        pass

    def explore_(self, observations):
        """Explore functionality: Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents """
    #    maybe call randomly explore or exploit to avoid bias towards the agent that generates observations
        return self.agent.exploit(observations)


    def exploit_(self, observations):
        """Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents"""

        return self.agent.exploit(observations)


    def exploit(self, observations):
        """Call either explore or exploit by chance to generate a diverse set of observations"""

        # print(type(observations[1][0]))
        # a = pickle.dump(observations[0], 'dumpfile')
        observ = []
        for obs_obj in observations[0]:
            observ.append(obs_obj)
        # print(observ[0], observ[0].hands, observ[0].parent_game)
        # print(observations[1][0][0])
        for i in range(len(observ)):
            
            self.c.execute('INSERT INTO obs VALUES (?, ?, ?)', (sqlite3.Binary(pickle.dumps(observ[i], pickle.HIGHEST_PROTOCOL)), sqlite3.Binary(pickle.dumps(observations[1][0][i], pickle.HIGHEST_PROTOCOL)),sqlite3.Binary(pickle.dumps(observations[1][1][i], pickle.HIGHEST_PROTOCOL))))
        self.conn.commit()

        self.c.execute('SELECT obs_vec, obs_act_vec FROM obs LIMIT 100')
        result = self.c.fetchall()
        obs_o = []
        act_vec = []
        for elem in result:
            act_vec.append(np.asarray(pickle.loads(elem[1])))
            obs_o.append(np.asarray(pickle.loads(elem[0])))
        # print('XXXXXXXXXX')
        obs_o = np.asarray(obs_o)
        act_vec = np.asarray(act_vec)
        # print(type(obs_o))
        # print(obs_o.shape)
        # print(observations[1][1].shape)
        # actio = np.asarray([observations[1][1][0], observations[1][1][1]])
        print(self.exploit_((observ[1],(obs_o, act_vec))))
        # self.exploit_((None,(obs_o, None)))
        # time.sleep(10)


        decide = random.randint(0,10) 
        if decide > 7:
          return self.explore_(observations)
        else:
          return self.exploit_(observations)
        

    # deprecated with new implementation for stacking
    def add_experience_first(self, observations, step_types):
        pass


    def add_experience(self, obersvation_tm1, actions, rewards, observations, step_types):
        '''Breaks input into chunks and adds them to experience buffer'''
        pass


    def shape_rewards(self, observations, moves):
        return (np.zeros(len(observations[1][0])), np.zeros(len(observations[1][0])))
        # return (0, 0)

    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.agent_params.history_size, 
                                            obs_len,
                                            n_states)

    def update(self):
        '''Train the agent's policy'''
        pass

    def requires_vectorized_observation(self):
        return True


    def save_weights(self, path, mean_reward):
        """saves the weights of all agents to the respective directories"""
        pass
    
    def save_obs_to_db(self, observation):
        pass

    def restore_weights(self, path):
        self.agent.restore_weights(path, path)



