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
# import ray
import pickle

"""
### Work in Progress###

An agent which maintains/manages and evaluates a population of DQN/Rainbow agents.

"""

Actions = np.ndarray
Observations = np.ndarray
Rewards = np.ndarray
LegalMoves = np.ndarray

# ray.init(num_cpus=8, ignore_reinit_error=True)

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


class AgentDQNPopulation:


    def __init__(
            self,
            n_states,
            env_obs_size,
            env_act_size,
            population_params: PBTParams = PBTParams(),
            agent_params: RlaxRainbowParams = RlaxRainbowParams(),
            reward_shaping_params: RewardShapingParams = RewardShapingParams()):

        self.pbt_params = population_params
        self.agent_params = agent_params
        self.reward_params = reward_shaping_params
        self.n_states = n_states
        self.pop_size = self.pbt_params.population_size
        self.env_act_size = env_act_size
        self.env_obs_size = observation_spec_vec_batch(env_obs_size, int(self.pop_size))[0]

        self.pop_size = self.pbt_params.population_size

        assert (self.n_states % self.pop_size) == 0, \
            "number of states must be a multiple of population size"
        self.obs_chunk_size = self.n_states // self.pop_size
        self.states_done = np.full((self.n_states,), False)
        self.states_reset = np.full((self.n_states,), False)
        self.evaluations = np.zeros((self.n_states,))

        # Make rewardshaping object as list/single objective for general/individual shaping
        if self.pbt_params.individual_reward_shaping:
            self.reward_shaper = []
            for i in range(self.pop_size):
                with gin.config_scope('agent'+str(i)):
                    self.reward_shaper.append(RewardShaper(self.reward_params))
        else:
            self.reward_shaper = RewardShaper(self.reward_params)

        '''
        1. If available: Load agents from checkpoints
        '''
        self.agents = []
        # TODO: --later-- start with weights -option
        # if start_with_weights is not None:
        #     print(start_with_weights)
        #     for aid, agent in enumerate(agents):
        #         if "agent_" + str(aid) in start_with_weights:
        #             agent.restore_weights(*(start_with_weights["agent_" + str(aid)]))

        '''
        2. Parse config files
        '''
        # # TODO: --done-- logic to generate actual DQN agents from params
        # if not self.pbt_params.agent_config_path:
        #     gin.parse_config_file(self.pbt_params.gin_files)
        #     gin_params = RlaxRainbowParams()
        #     print(gin_params)
        # # if not self.params.agent_config_path:
        # #     if len(self.params.gin_files) == 1:
        # #         gin.parse_config_file(self.params.gin_files)
        # #         agent_params = [RlaxRainbowParams()]
        # #         print(agent_params)
        # #     # TODO: ---done--- implement logic for different gins upon start -
        # #     else:
        # #         assert (len(self.params.gin_files) == self.pop_size, \
        # #                 "number of gin_files(agent config params) doesn't match pop_size!")
        # #
        # #         for i in range(self.pop_size):
        # #             gin.parse_config_file(self.params.gin_files[i])
        # #             agent_params = []
        # #             agent_params.append = RlaxRainbowParams()

        '''
        3. Initialize atomic agents with randomly chosen lr and buffersize from given intervals
        '''
        def sample_buffersize():
            exp_factor = math.log(self.agent_params.experience_buffer_size, 2)
            buffer_sizes_start = [2**i for i in range(int(exp_factor) - self.pbt_params.buffersize_start_factor, \
                                                        int(exp_factor) + self.pbt_params.buffersize_start_factor)]                                     
            return random.choice(buffer_sizes_start)

        def sample_init_lr():
            min_lr = self.agent_params.learning_rate - self.pbt_params.lr_start_value
            max_lr = self.agent_params.learning_rate + self.pbt_params.lr_start_value
            return random.uniform(min_lr, max_lr)

        agent_configs = []
        for j in range(self.pop_size):
            custom_params = self.agent_params
            custom_params = custom_params._replace(learning_rate = sample_init_lr())
            custom_params = custom_params._replace(experience_buffer_size = sample_buffersize())
            agent = DQNAgent(self.env_obs_size,
                             self.env_act_size,
                             custom_params)
            self.agents.append(agent)
        agents_status(self.agents)

        self.evo_steps = 0


    def explore(self, observations):
        """Explore functionality: Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents """
        # TODO: concatenate HanabiMoveVectors / slicing HanabiObservationVector
        action_chunks = []
        for i, agent in enumerate(self.agents):
            obs_chunk = (observations, (observations[1][0][i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size],
                         observations[1][1][i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size]))
            action_chunks.append(agent.explore(obs_chunk))
        actions = np.concatenate(action_chunks, axis = 0)
        return actions


    def exploit(self, observations):
        """Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents"""
        chunk_size = int(len(observations[0])/self.pop_size)
        action_chunks = []
        for i, agent in enumerate(self.agents):
            obs_chunk = (observations, (observations[1][0][i * chunk_size : (i + 1) * chunk_size],
                         observations[1][1][i * chunk_size : (i + 1) * chunk_size]))
            action_chunks.append(agent.exploit(obs_chunk))
        actions = np.concatenate(action_chunks, axis = 0)
        return actions

    # deprecated with new implementation for stacking
    def add_experience_first(self, observations, step_types):
        pass



    def add_experience(self, obersvation_tm1, actions, rewards, observations, step_types):
        '''Breaks input into chunks and adds them to experience buffer'''
        for i, agent in enumerate(self.agents):
            obs_tm1_chunk = (obersvation_tm1, (obersvation_tm1[1][0][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size],
                                        obersvation_tm1[1][1][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]))
            obs_chunk = (observations, (observations[1][0][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size],
                                        observations[1][1][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]))
            action_chunk = actions[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]
            reward_chunk = rewards[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]
            step_chunk = step_types[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]

            agent.add_experience(obs_tm1_chunk, action_chunk, reward_chunk, obs_chunk, step_chunk)

    def shape_rewards(self, observations, moves):
    
        if self.reward_shaper is not None:
            if type(self.reward_shaper) == list:
                #TODO can this be sliced directly?
                observations_ = [observation for observation in observations[0]]
                shaped_rewards, shape_type = [], []

                for i, shaper in enumerate(self.reward_shaper):
                    s_rew, t_sha = shaper.shape(observations_[i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size], \
                                                    moves[i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size])
                    shaped_rewards.extend(s_rew)
                    shape_type.extend(t_sha)

            else:
                shaped_rewards, shape_type = self.reward_shaper.shape(observations[0], 
                                                                    moves)
            return np.array(shaped_rewards), np.array(shape_type)
        return (0, 0)

    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.agent_params.history_size, 
                                            obs_len,
                                            n_states)

    def update(self):
        '''Train the agent's policy'''
        for agent in self.agents:
            agent.update()


    def requires_vectorized_observation(self):
        return True


    def save_weights(self, path):
        """saves the weights of all agents to the respective directories"""
        for i, agent in enumerate(self.agents):
            agent.save_weights(os.path.join(path, "agent_" + str(i)), "ckpt_" + str(agent.train_step))



    def survival_fittest(self, index_survivors, index_losers):
        """Copies the network weights from those agents considered survivors over to those not considered and samples new
            parameters for those overwritten (learning rate / buffersize)"""

        survivor_attributes = []
        print('survivor_attributes is ', len(survivor_attributes))
        for survivor in index_survivors:
            survivor_attributes.append(self.agents[survivor].get_agent_attributes())
        for loser in index_losers:
            winner = random.choice(survivor_attributes)
            self.agents[loser].overwrite_weights(winner[2])
            self.agents[loser].overwrite_lr(self.pbt_params.lr_factor, winner[0])
            self.agents[loser].change_buffersize(self.pbt_params.buffersize_factor, winner[1])
        agents_status(self.agents)
