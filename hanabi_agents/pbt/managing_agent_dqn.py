from typing import Callable
import numpy as np
import math
from hanabi_learning_environment import pyhanabi_pybind as pyhanabi
from hanabi_agents.rlax_dqn import DQNAgent, DQNParallel, RlaxRainbowParams, PBTParams
from hanabi_agents.rlax_dqn import RewardShapingParams, RewardShaper
from hanabi_agents.rlax_dqn import VectorizedObservationStacker
import gin
from dm_env import specs as dm_specs
from typing import Tuple, Dict, List, Union
import random
import os
from dm_env import TimeStep, StepType
import pickle
import sqlite3
import time
from math import e
import names
from sklearn import metrics

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
            reward_shaping_params: RewardShapingParams = RewardShapingParams(),
            agent_data = None,
            eval_run = False):

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
        self.eval_run = eval_run
        self.tmp_obs = []
        self.pbt_history = []
        self.pbt_history_params = []

        self.max_score = 15 #"""!!!!!!!!!!!!!!!!!!!!!!!!!"""
        

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
        3. Initialize atomic agents with randomly chosen lr and buffersize from given intervals
        '''
        def sample_buffersize():
            exp_factor = math.log(self.agent_params.experience_buffer_size, 2)
            buffer_sizes_start = [2**i for i in range(int(exp_factor) - self.pbt_params.buffersize_start_factor, \
                                                        int(exp_factor) + self.pbt_params.buffersize_start_factor)]                                     
            return random.choice(buffer_sizes_start)

        def sample_init_lr():
            return random.choice(np.linspace(self.pbt_params.lr_min, self.pbt_params.lr_max, self.pbt_params.lr_sample_size))
        
        def sample_init_alpha():
            return random.choice(np.linspace(self.pbt_params.alpha_min, self.pbt_params.alpha_max, self.pbt_params.alpha_sample_size))
        

        agent_configs = []

        for j in range(self.pop_size):
            if not agent_data:
                print('Sampling new Agent!')
                custom_params = self.agent_params
                custom_params = custom_params._replace(learning_rate = sample_init_lr())
                custom_params = custom_params._replace(experience_buffer_size = sample_buffersize())
                custom_params = custom_params._replace(priority_w = sample_init_alpha())
                #alpha
                agent = DQNParallel(self.env_obs_size,
                                        self.env_act_size,
                                        custom_params)
                self.agents.append(agent)
            else:
                print('New Agent from evolved Parameters!')
                custom_params = self.agent_params
                custom_params = custom_params._replace(learning_rate = agent_data[j]['lr'])
                custom_params = custom_params._replace(experience_buffer_size = agent_data[j]['buffersize'])
                custom_params = custom_params._replace(priority_w = agent_data[j]['alpha'])
                agent = DQNParallel(self.env_obs_size,
                        self.env_act_size,
                        custom_params)
                self.agents.append(agent)
        
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
        if self.eval_run:
            self.tmp_obs.append(observations)
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
    
    def save_weights_fast(self, path, mean_reward):
        """saves the weights of all agents to the respective directories"""
        os.makedirs(path)
        for i, agent in enumerate(self.agents):
            agent.save_weights(path, "ckpt_" + str(agent.train_step) + '_agent_' + str(i))


    def save_specific_agent(self, path, index, act_vec, score):
        if not os.path.isdir(path):
            os.makedirs(path)
        if not os.path.isdir(os.path.join(path, 'agents')):
            os.makedirs(os.path.join(path, 'agents'))
        """saves the weights of specific agents that meet a certain condition"""
        agent_name = names.get_full_name().replace(' ', '_') 
        parameters_name = agent_name + '_lr_buf_alpha_w_play_disc'
        weights_name = agent_name + '_{}'.format(str(score).replace('.','_'))
        self.agents[index].save_weights(os.path.join(path, 'agents'), weights_name)
        path_npy = os.path.join(path, 'action_vecs/{}'.format(agent_name + '_action_vector'))
        np.save(path_npy, act_vec)
        np.save(os.path.join(path, 'agents/{}'.format(parameters_name)), np.asarray([self.agents[index].learning_rate, 
                    self.agents[index].buffersize, 
                    self.agents[index].experience.alpha,
                    self.reward_shaper[index].w_play_probability,
                    self.reward_shaper[index].min_play_probability,
                    self.reward_shaper[index].penalty_last_of_kind]))

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
    
    def generate_action_vectors(self, obs):
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

    @staticmethod
    def cosine_distance(actions_agent_a, actions_agent_b, no_obs):
        pass

    
    def mutual_information(self, actions_agent_a, actions_agent_b, no_obs):
        '''Compares both vectors by calculating the mutual information'''
        return 1 - metrics.normalized_mutual_info_score(actions_agent_a, actions_agent_b)

    @staticmethod
    def simple_match(actions_agent_a, actions_agent_b, no_obs):
        '''Compares both action vectors and calculates the number of matching actions'''
        return (1 - np.sum(actions_agent_a == actions_agent_b)/no_obs)

    def measure_diversity(self, db_path, function = mutual_information):
        if self.pbt_params.use_db:
            obs = self._load_db(db_path)
        else:
            obs = np.concatenate(self.tmp_obs)
        actions_curr_agents, agent_indices = self.generate_action_vectors(obs)
        actions_exis_agents = self.load_existing_agents()
        actions_curr_agents.extend(actions_exis_agents)
        combined = actions_curr_agents
        no_obs = len(combined[0])
        diversity_matrix = [[] for i in range(len(combined))]
        for i, elem in enumerate(combined):
            for vec in combined:
                mut_info = self.mutual_information(elem, vec, no_obs)
                diversity_matrix[i].append(mut_info)
        diversity_matrix = np.asarray(diversity_matrix)
        np.fill_diagonal(diversity_matrix, 1)
        print('Diversity matrix equals: \n {}'.format(diversity_matrix))
        return diversity_matrix, actions_curr_agents


    def quantify_diversity(self, diversity_mat, mean_reward):
        minima = np.min(diversity_mat, axis = 1)
        #linear influence of diversity measure
        # mean_reward[self.readiness] = mean_reward[self.readiness] + mean_reward[self.readiness]/self.max_score * 5 * minima[:np.sum(self.readiness)]     
        #sigmoid activated influence
        mean_reward[self.readiness] = mean_reward[self.readiness] + 1/(1 + e**-(mean_reward[self.readiness] - 2/3*self.max_score)) * self.pbt_params.w_diversity * minima[:np.sum(self.readiness)]
        return mean_reward
    
    def pbt_log(self, index_overwriting, index_overwritten):
        if not self.pbt_history:
            new_row = list(range(self.pop_size))
        else:
            new_row = self.pbt_history[-1][0].copy()
        which = [False for i in range(self.pop_size)]
        for i, index in enumerate(index_overwritten):
            new_row[index] = index_overwriting[i]
            which[index] = True
        self.pbt_history.append([new_row, which])

    def pbt_log_params(self):
        new_row = []
        new_agent = []
        for i, agent in enumerate(self.agents):
            new_agent.append(agent.learning_rate)
            new_agent.append(agent.buffersize)
            new_agent.append(agent.experience.alpha)
            new_agent.append(self.reward_shaper[i].penalty_last_of_kind)
            new_agent.append(self.reward_shaper[i].w_play_probability)
            new_agent.append(self.reward_shaper[i].min_play_probability)
            new_row.append(new_agent)
        self.pbt_history_params.append(new_row)


    def pbt_eval(self, mean_reward, output_dir):
        """Copies the network weights from those agents considered survivors over to those not considered and samples new
            parameters for those overwritten (learning rate / buffersize)"""
        self.readiness = self.pbt_counter >= self.pbt_params.life_span
        print('Mean reward without Diversity: {}'.format(mean_reward))
        diversity_matrix, action_vectors = self.measure_diversity( self.pbt_params.db_path)
        mean_reward_div = self.quantify_diversity(diversity_matrix, mean_reward)
        print('Mean reward + Diversity measure is {}'.format(mean_reward_div))
        if np.max(mean_reward_div[self.readiness]) >= self.pbt_params.saver_threshold:
            print('XXXXXXXXXXXXXXXXXXXXXXXXXXXX SAVING A NEW AGENT XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
            index_max = np.where(self.readiness == 1)[0][np.argmax(mean_reward_div[self.readiness])]
            score = np.max(mean_reward_div[self.readiness])
            self.save_specific_agent(self.pbt_params.pool_path, index_max, action_vectors[np.argmax(mean_reward_div[self.readiness])], score)

        index_survivor, index_losers = self._choose_fittest(mean_reward_div)
        survivor_attributes = []
        survivor_dominant = []
        for survivor in index_survivor:
            survivor_dominant.append(random.choice(list(range(len(index_survivor)))))
            survivor_attributes.append(self.agents[survivor].get_agent_attributes())
        index_dominant = []
        for elem in survivor_dominant:
            index_dominant.append(index_survivor[elem])
        self.pbt_log(index_dominant, index_losers)
        for i, loser in enumerate(index_losers):
            winner = survivor_attributes[survivor_dominant[i]]
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
        self.pbt_log_params()

        # agents_status(self.agents)
        print('Update current epochs per agent {}'.format(self.pbt_counter))
        time.sleep(10)

    def save_pbt_log(self, path, epoch_circle):
        path = os.path.join(path, 'pbt_log_data')
        if not os.path.isdir(path):
            os.makedirs(path)
        print('before saving', self.pbt_history)
        pbt_name = path + 'pbt_history_generation_{}'.format(epoch_circle)
        path_pbt_hist = path + 'pbt_params_history_generation_{}'.format(epoch_circle)
        a = np.asarray(self.pbt_history)
        b = np.asarray(self.pbt_history_params)
        np.save(pbt_name, a)
        np.save(path_pbt_hist, b)


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
            parameters[0].append(float(agent.learning_rate))
            parameters[1].append(int(agent.buffersize))
            parameters[2].append(int(agent.train_step))
            parameters[3].append(float(self.reward_shaper[i].penalty_last_of_kind))
            parameters[4].append(float(self.reward_shaper[i].min_play_probability))
            parameters[5].append(float(self.reward_shaper[i].w_play_probability))  
            if self.agent_params.use_priority:
                values_sumtree = agent.experience.get_info()
                buffer = agent.experience
                buffer.sum_tree = None
                experience.append((buffer, values_sumtree))
            else:
                experience.append(agent.experience)


        return {'online_weights' : online_weights, 'trg_weights' : trg_weights,
                'opt_states' : opt_states, 'experience' : experience, 'parameters' : parameters}
    
    def save_min_characteristics(self):
        characteristics = {'buffersize' : [], 'lr' : [], 'alpha': []}
        for agent in self.agents:
            print('aaaagent')
            characteristics['buffersize'].append(agent.buffersize)
            characteristics['lr'].append(agent.learning_rate)
            characteristics['alpha'].append(agent.experience[0].alpha)
        return characteristics

    def restore_characteristics(self, characteristics):
        for i, agent in enumerate(self.agents):
            print('Restoring agent no {}!'.format(i))
            agent.online_params = characteristics['online_weights'][i]
            agent.trg_params = characteristics['trg_weights'][i]
            agent.opt_state = characteristics['opt_states'][i]            
            agent.learning_rate = characteristics['parameters'][0][i]
            agent.buffersize = characteristics['parameters'][1][i]
            agent.train_step = characteristics['parameters'][2][i]
            self.reward_shaper[i].penalty_last_of_kind = characteristics['parameters'][3][i]
            self.reward_shaper[i].min_play_probability = characteristics['parameters'][4][i]
            self.reward_shaper[i].w_play_probability = characteristics['parameters'][5][i]
            if self.agent_params.use_priority:
                agent.experience = characteristics['experience'][i][0]
                agent.experience.restore_sumtree(characteristics['experience'][i][1]) 
            else:
                agent.experience = characteristics['experience'][i] 

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