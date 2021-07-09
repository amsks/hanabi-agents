from typing import Callable
import numpy as np
import math
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from hanabi_agents.rlax_dqn import DQNAgent, RlaxRainbowParams, PBTParams
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper

import gin
from dm_env import specs as dm_specs
from typing import Tuple, Dict, List, Union
import random
import os
from dm_env import TimeStep, StepType

import time
import pickle
import sqlite3




class DataGenerationAgent:

    def __init__(
            self,
            n_states,
            env_obs_size,
            env_act_size,
            agent_params: RlaxRainbowParams = RlaxRainbowParams(),
            db_path = 'obs.db'):

        self.agent_params = agent_params
        self.n_states = n_states
        self.env_act_size = env_act_size
        self.env_obs_size = env_obs_size
        self.reward_shaper = None


        self.states_done = np.full((self.n_states,), False)
        self.states_reset = np.full((self.n_states,), False)
        self.evaluations = np.zeros((self.n_states,))

        self.conn = sqlite3.connect(db_path)
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

        self.agent = DQNAgent(observation_spec = self.env_obs_size,
                                action_spec = self.env_act_size,
                                params = agent_params)


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
        observ = []
        #HanabiObservationsVector not iterable yet:
        for obs_obj in observations[0]:
            observ.append(obs_obj)
        #save observation to DB
        for i in range(len(observ)):          
            self.c.execute('INSERT INTO obs VALUES (?, ?, ?)', (sqlite3.Binary(pickle.dumps(observ[i], pickle.HIGHEST_PROTOCOL)),
                                                                sqlite3.Binary(pickle.dumps(observations[1][0][i], pickle.HIGHEST_PROTOCOL)),
                                                                sqlite3.Binary(pickle.dumps(observations[1][1][i], pickle.HIGHEST_PROTOCOL)))) 
        self.conn.commit()

        #Take next step in all parallel states by exploit or explore policy
        decide = random.randint(0,10) 
        if decide > 7:
          return self.explore_(observations)
        else:
          return self.exploit_(observations)

        ### HOW TO RECOVER OBSERVATIONS FROM DB### 
        # self.c.execute('SELECT obs_vec, obs_act_vec FROM obs LIMIT 100')
        # result = self.c.fetchall()
        # obs_o = []
        # act_vec = []
        # for elem in result:
        #     act_vec.append(np.asarray(pickle.loads(elem[1])))
        #     obs_o.append(np.asarray(pickle.loads(elem[0])))
        # obs_o = np.asarray(obs_o)
        # act_vec = np.asarray(act_vec)
        
    def restore_weights(self, path):
        self.agent.restore_weights(path, path)



    # all unnecessary functions below but part of functions for every agent
    def add_experience(self, obersvation_tm1, actions, rewards, observations, step_types):
        '''Breaks input into chunks and adds them to experience buffer'''
        pass


    def shape_rewards(self, observations, moves):
        return (np.zeros(len(observations[1][0])), np.zeros(len(observations[1][0])))


    def create_stacker(self, obs_len, n_states):
        pass

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




