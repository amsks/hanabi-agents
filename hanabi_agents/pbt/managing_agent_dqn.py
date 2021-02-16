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
import ray
import pickle
import sqlite3
import time
from math import e
import names

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
        self.prev_reward = 0
        self.pbt_counter = np.zeros(self.pop_size)

        self.max_score = 15 #"""!!!!!!!!!!!!!!!!!!!!!!!!!"""

        # Make rewardshaping object as list/single objective for general/individual shaping
        # if self.pbt_params.individual_reward_shaping:
        #     self.reward_shaper = []
        #     # for i in range(self.pop_size):
        #     with gin.config_scope('agent'+str('_0')):
        #         self.reward_shaper.append(RewardShaper(self.reward_params))
        # else:
        #     self.reward_shaper = []
        #     for i in range(self.pop_size):
        #         self.reward_shaper.append(RewardShaper(self.reward_params))
        # if self.reward_params.use_reward_shaping:
        self.reward_shaper = []
        for i in range(self.pop_size):
            self.reward_shaper.append(RewardShaper(self.reward_params))
        # else:
        #     self.reward_shaper = None
        
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
            # agent = DQNAgent(self.env_obs_size,
            #                  self.env_act_size,
            #                  custom_params)
            self.agents.append(agent)
        # agents_status(self.agents)

        self.evo_steps = 0
        print('No of agents in this', len(self.agents))

    def explore(self, observations):
        """Explore functionality: Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents """
        action_chunks = []
        for i, agent in enumerate(self.agents):
            obs_chunk = (observations, (observations[1][0][i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size],
                         observations[1][1][i * self.obs_chunk_size : (i + 1) * self.obs_chunk_size]))
            actions_rem = agent.explore(obs_chunk)             
            action_chunks.append(actions_rem)
        actions = np.concatenate(action_chunks, axis = 0)
        return actions


    def exploit(self, observations):
        """Breaks env_obs in chunks of obs_chunk_size and passes it to respective sub-agents"""
        chunk_size = int(len(observations[0])/self.pop_size)
        action_chunks = []
        for i, agent in enumerate(self.agents):
            
            obs_chunk = (observations, (observations[1][0][i * chunk_size : (i + 1) * chunk_size],
                         observations[1][1][i * chunk_size : (i + 1) * chunk_size]))
            actions_rem = agent.exploit(obs_chunk)
            action_chunks.append(actions_rem)
        actions = np.concatenate(action_chunks, axis = 0)

        return actions

    # deprecated with new implementation for stacking
    def add_experience_first(self, observations, step_types):
        pass



    def add_experience(self, obersvation_tm1, actions, rewards, observations, step_types):
        '''Breaks input into chunks and adds them to experience buffer'''
        # start_time = time.time()
        for i, agent in enumerate(self.agents):
            obs_tm1_chunk = (obersvation_tm1, (obersvation_tm1[1][0][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size],
                                        obersvation_tm1[1][1][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]))
            obs_chunk = (observations, (observations[1][0][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size],
                                        observations[1][1][i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]))
            action_chunk = actions[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]
            reward_chunk = rewards[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]
            step_chunk = step_types[i * self.obs_chunk_size: (i + 1) * self.obs_chunk_size]

            agent.add_experience(obs_tm1_chunk, action_chunk, reward_chunk, obs_chunk, step_chunk)
        # print('add_exp took {} seconds'.format(time.time()-start_time))


    def shape_rewards(self, observations, moves):
    
        if self.reward_params.use_reward_shaping:
        # if type(self.reward_shaper) == list:
        #TODO can this be sliced directly?
            observations_ = [observation for observation in observations[0]]
            no_states = int(len(observations_) / self.pop_size)
            moves_ = [move for move in moves]
            shaped_rewards, shape_type = [], []
            for i, shaper in enumerate(self.reward_shaper):
            
                s_rew, t_sha = shaper.shape(observations_[i * no_states : (i + 1) * no_states], \
                                                moves_[i * no_states : (i + 1) * no_states])
                shaped_rewards.extend(s_rew)
                shape_type.extend(t_sha)
            return np.array(shaped_rewards), np.array(shape_type)
        return (np.zeros(len(observations[1][0])), np.zeros(len(observations[1][0])))

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


    def save_weights(self, path, mean_reward):
        """saves the weights of all agents to the respective directories"""
        for i, agent in enumerate(self.agents):
            if self.prev_reward < mean_reward[i]:
                agent.save_weights(path, "ckpt_" + str(agent.train_step))
                self.prev_reward = mean_reward[i]
    
    def save_specific_agent(self, path, index, act_vec, score):
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(os.path.join(path, 'agents')):
            os.makedirs(os.path.join(path, 'agents'))
        """saves the weights of specific agents that meet a certain condition"""
        agent_name = names.get_full_name().replace(' ', '_') + '_{}'.format(str(score).replace('.','_'))
        self.agents[index].save_weights(os.path.join(path, 'agents'), agent_name)
        path_npy = os.path.join(path, 'action_vecs/{}'.format(agent_name))
        np.save(path_npy, act_vec)

    def _choose_fittest(self, mean_reward):
        """Chosses the fittest agents after evaluation run and overwrites all the other agents with weights + permutation of lr + buffersize"""
        rdy_agents = np.sum(self.readiness)
        orig_index = np.where(self.readiness == 1)[0]
        no_fittest = rdy_agents - int(rdy_agents * self.pbt_params.discard_percent)
        index_loser = orig_index[np.argpartition(mean_reward[self.readiness], no_fittest)[:no_fittest]]
        index_survivor = orig_index[np.argpartition(-mean_reward[self.readiness], no_fittest)[:no_fittest]]
        print('Agents to pass their weights are {}'.format(index_survivor))
        print('Agents to be overwritten are {}'.format(index_loser))
        print('Current epochs per agent {}'.format(self.pbt_counter))
        time.sleep(10)
        return index_survivor, index_loser

    def _load_db(self, path):
        conn = sqlite3.connect(path)
        c = conn.cursor()

        if self.requires_vectorized_observation:
            c.execute('SELECT obs_vec, obs_act_vec FROM obs LIMIT {}'.format(self.pbt_params.obs_no))
            query = c.fetchall()
            obs_o = []
            act_vec = []
            for elem in query:
                act_vec.append(np.asarray(pickle.loads(elem[1])))
                obs_o.append(np.asarray(pickle.loads(elem[0])))
            return (None,(np.asarray(obs_o), np.asarray(act_vec)))
        else:
            c.execute('SELECT obs_obj FROM obs LIMIT {}'.format(self.pbt_params.obs_no))
            query = c.fetchall()
            obs = pickle.loads(query[0])
            return obs
    
    def generate_action_vector(self, obs):
        action_vec = []
        agent_indices = np.where(self.readiness == True)[0].tolist()
        for elem in agent_indices:
            action_vec.append(self.agents[int(elem)].exploit(obs))
        return action_vec, agent_indices

    def load_existing_agents(self):
        vecs_path = os.path.join(self.pbt_params.pool_path, 'action_vecs')
        
        if not os.path.isdir(vecs_path):
            os.makedirs(vecs_path)
        no_agents = len([name for name in os.listdir(vecs_path) if os.path.isfile(name)])
        print(vecs_path, no_agents)
        action_vecs = []      
        for file in os.listdir(vecs_path):
            action_vecs.append(np.load(os.path.join(vecs_path, file)))
        return action_vecs

    def mutual_information(self, db_path):
        obs = self._load_db(db_path)
        actions_curr_agents, agent_indices = self.generate_action_vector(obs)
        actions_exis_agents = self.load_existing_agents()
        actions_curr_agents.extend(actions_exis_agents)
        combined = actions_curr_agents
        no_obs = len(combined[0])
        diversity_matrix = [[] for i in range(len(combined))]
        for i, elem in enumerate(combined):
            for vec in combined:
                mut_info = (1 - np.sum(elem == vec)/no_obs)
                diversity_matrix[i].append(mut_info)
        diversity_matrix = np.asarray(diversity_matrix)
        np.fill_diagonal(diversity_matrix, 1)
        print('Diversity matrix equals: \n {}'.format(diversity_matrix))
        return diversity_matrix, actions_curr_agents

    def cosine_distance(self):
        pass

    def quantify_diversity(self, diversity_mat, mean_reward):
        minima = np.min(diversity_mat, axis = 1)
        #linear influence of diversity measure
        # mean_reward[self.readiness] = mean_reward[self.readiness] + mean_reward[self.readiness]/self.max_score * 5 * minima[:np.sum(self.readiness)]     
        #sigmoid activated influence
        mean_reward[self.readiness] = mean_reward[self.readiness] + 1/(1 + e**-(mean_reward[self.readiness] - 2/3*self.max_score)) * self.pbt_params.w_diversity * minima[:np.sum(self.readiness)]
        return mean_reward
    

    def pbt_eval(self, mean_reward, output_dir):
        """Copies the network weights from those agents considered survivors over to those not considered and samples new
            parameters for those overwritten (learning rate / buffersize)"""
        self.readiness = self.pbt_counter >= self.pbt_params.life_span
        print('Mean reward without Diversity: {}'.format(mean_reward))
        diversity_matrix, action_vectors = self.mutual_information( self.pbt_params.db_path)
        mean_reward_div = self.quantify_diversity(diversity_matrix, mean_reward)
        print('Mean reward + Diversity measure is {}'.format(mean_reward_div))

        if np.max(mean_reward_div[self.readiness]) >= self.pbt_params.saver_threshold:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX SAVING A NEW AGENT XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            index_max = np.where(self.readiness == 1)[0][np.argmax(mean_reward_div[self.readiness])]
            score = np.max(mean_reward_div[self.readiness])
            self.save_specific_agent(self.pbt_params.pool_path, index_max, action_vectors[np.argmax(mean_reward_div[self.readiness])], score)

        index_survivor, index_losers = self._choose_fittest(mean_reward_div)

        survivor_attributes = []
        for survivor in index_survivor:
            survivor_attributes.append(self.agents[survivor].get_agent_attributes())
        for loser in index_losers:
            winner = random.choice(survivor_attributes)
            self.agents[loser].overwrite_weights(winner[2])
            if self.pbt_params.change_learning_rate:
                self.agents[loser].overwrite_lr(self.pbt_params.lr_factor, winner[0])
            if self.pbt_params.change_buffersize:
                self.agents[loser].change_buffersize(self.pbt_params.buffersize_factor, winner[1])
            if self.reward_params.use_reward_shaping:
                if self.pbt_params.change_min_play_probability:
                    new_val = random.choice([-self.pbt_params.min_play_probability_pbt, 0 , self.pbt_params.min_play_probability_pbt])
                    self.reward_shaper[loser].change_min_play_prob(new_val)
                if self.pbt_params.change_w_play_probability:
                    new_val = random.choice([-self.pbt_params.w_play_probability_pbt, 0, self.pbt_params.w_play_probability_pbt])
                    self.reward_shaper[loser].change_w_play_probability(new_val)
                    
                if self.pbt_params.change_penalty_last_of_kind:
                    new_val = random.choice([-self.pbt_params.penalty_last_of_kind_pbt, 0, self.pbt_params.penalty_last_of_kind_pbt])
                    self.reward_shaper[loser].change_penalty_last_kind(new_val)
                
            self.pbt_counter[loser] = 0
        # agents_status(self.agents)
        print('Update current epochs per agent {}'.format(self.pbt_counter))
        time.sleep(10)

    def save_pbt_log(self):
        pass


    def save_characteristics(self):
        online_weights = []
        trg_weights = []
        opt_states = []
        experience = []
        parameters = [[],[],[],[],[],[]]
        for i, agent in enumerate(self.agents):
            online_weights.append(agent.online_params)
            trg_weights.append(agent.trg_params)
            opt_states.append(agent.opt_state)
            experience.append(agent.experience)
            parameters[0].append(float(agent.learning_rate))
            parameters[1].append(int(agent.buffersize))
            parameters[2].append(int(agent.train_step))
            parameters[3].append(float(self.reward_shaper[i].penalty_last_of_kind))
            parameters[4].append(float(self.reward_shaper[i].min_play_probability))
            parameters[5].append(float(self.reward_shaper[i].w_play_probability))        

        return {'online_weights' : online_weights, 'trg_weights' : trg_weights,
                'opt_states' : opt_states, 'experience' : experience, 'parameters' : parameters}
    
    def restore_characteristics(self, characteristics):
        for i, agent in enumerate(self.agents):
            print('Restoring agent no {}!'.format(i))
            agent.online_params = characteristics['online_weights'][i]
            agent.trg_params = characteristics['trg_weights'][i]
            agent.opt_state = characteristics['opt_states'][i]      
            agent.experience = characteristics['experience'][i]       
            agent.learning_rate = characteristics['parameters'][0][i]
            agent.buffersize = characteristics['parameters'][1][i]
            agent.train_step = characteristics['parameters'][2][i]
            self.reward_shaper[i].penalty_last_of_kind = characteristics['parameters'][3][i]
            self.reward_shaper[i].min_play_probability = characteristics['parameters'][4][i]
            self.reward_shaper[i].w_play_probability = characteristics['parameters'][5][i]
            print('Agent_{} has lr {}, BufSiz {}, TrainStep {}, PenLastKind{}, MinPlayProb {}, wPlayProb {}'.format(i, 
                                                                                                                    agent.learning_rate, 
                                                                                                                    agent.buffersize, 
                                                                                                                    agent.train_step, 
                                                                                                                    self.reward_shaper[i].penalty_last_of_kind, 
                                                                                                                    self.reward_shaper[i].min_play_probability, 
                                                                                                                    self.reward_shaper[i].w_play_probability))
    def restore_weights(self, restore_path):
        for agent in self.agents:
            agent.restore_weights(restore_path, restore_path)
    
    def increase_pbt_counter(self):
        self.pbt_counter += 1