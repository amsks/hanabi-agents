"""
This file implements a DQNAgent.
"""
import collections
import pickle
from functools import partial
from typing import Tuple, List
from os.path import join as join_path

import numpy as onp

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import rlax
import chex

from .experience_buffer import ExperienceBuffer
from .priority_buffer import PriorityBuffer
from .noisy_mlp import NoisyMLP
from .params import RlaxRainbowParams
from .vectorized_stacker import VectorizedObservationStacker

DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy"])


class DQNPolicy:
    """greedy and epsilon-greedy policies for DQN agent"""

    @staticmethod
    def _categorical_sample(key, probs):
        """Sample from a set of discrete probabilities."""
        cpi = jnp.cumsum(probs, axis=-1)
        # TODO
        # sometimes illegal actions emerge due to numerical inaccuracy.
        # e.g. 2 actions, last action 100%: -> cpi = [0, 1]
        # but due to numerical stuff: cpi = [0, 0,997]
        # sample rnd = 0.999 -> rnd > cpi = [T, T] -> argmin returns 0 instead of 1
        cpi = jax.ops.index_update(cpi, jax.ops.index[:, -1], 1.)
        rnds = jax.random.uniform(key, shape=probs.shape[:-1] + (1,), maxval=0.999)
        return jnp.argmin(rnds > cpi, axis=-1)

    @staticmethod
    def _mix_with_legal_uniform(probs, epsilon, legal):
        """Mix an arbitrary categorical distribution with a uniform distribution."""
        num_legal = jnp.sum(legal, axis=-1, keepdims=True)
        uniform_probs = legal / num_legal
        return (1 - epsilon) * probs + epsilon * uniform_probs

    @staticmethod
    def _argmax_with_random_tie_breaking(preferences):
        """Compute probabilities greedily with respect to a set of preferences."""
        optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
        return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)

    @staticmethod
    def legal_epsilon_greedy(epsilon=None):
        """An epsilon-greedy distribution with illegal probabilities set to zero"""

        def sample_fn(key: chex.Array,
                      preferences: chex.Array,
                      legal: chex.Array,
                      epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return DQNPolicy._categorical_sample(key, probs)

        def probs_fn(preferences: chex.Array, legal: chex.Array, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            return DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)

        def logprob_fn(sample: chex.Array,
                       preferences: chex.Array,
                       legal: chex.Array,
                       epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return rlax.base.batched_index(jnp.log(probs), sample)

        def entropy_fn(preferences: chex.Array, legal: chex.Array, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return -jnp.nansum(probs * jnp.log(probs), axis=-1)

        return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def policy(
            network,
            use_distribution: bool,
            atoms,
            net_params,
            epsilon: float,
            key: float,
            obs: chex.Array,
            lms: chex.Array):
        """Sample action from epsilon-greedy policy.

        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q values
        # calculate q value from distributional output
        # by calculating mean of distribution
        if use_distribution:
            logits = network.apply(net_params, None, obs)
            probs = jax.nn.softmax(logits, axis=-1)
            q_vals = jnp.mean(probs * atoms, axis=-1)
            
        # q values equal network output
        else:
            q_vals = network.apply(net_params, None, obs)
        
        # mask q values of illegal actions
        q_vals = jnp.where(lms, q_vals, -jnp.inf)

        # compute actions
        actions = DQNPolicy.legal_epsilon_greedy(epsilon=epsilon).sample(key, q_vals, lms)
        return q_vals, actions

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def eval_policy(
            network,
            use_distribution: bool,
            atoms,
            net_params,
            key,
            obs: chex.Array,
            lms: chex.Array):
        """Sample action from greedy policy.
        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q values
        # calculate q value from distributional output
        # by calculating mean of distribution
        if use_distribution:
            logits = network.apply(net_params, None, obs)
            probs = jax.nn.softmax(logits, axis=-1)
            q_vals = jnp.mean(probs * atoms, axis=-1)
            
        # q values equal network output
        else:
            q_vals = network.apply(net_params, None, obs)
        
        # mask q values of illegal actions
        q_vals = jnp.where(lms, q_vals, -jnp.inf)

        # select best action
        return rlax.greedy().sample(key, q_vals)

class DQNLearning:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2, 10, 11))
    def update_q(network, atoms, optimizer, online_params, trg_params, opt_state,
                 transitions, discount_t, prios, beta_is, use_double_q, use_distribution): 
        """Update network weights wrt Q-learning loss.

        Args:
            network    -- haiku Transformed network.
            optimizer  -- optimizer.
            net_params -- parameters (weights) of the network.
            opt_state  -- state of the optimizer.
            q_tm1      -- q-value of state-action at time t-1.
            obs_tm1    -- observation at time t-1.
            a_tm1      -- action at time t-1.
            r_t        -- reward at time t.
            term_t     -- terminal state at time t?
        """

        # calculate double q td loss for distributional network
        def categorical_double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            q_logits_tm1 = network.apply(online_params, None, obs_tm1)
            q_logits_t = network.apply(trg_params, None, obs_t)
            q_logits_sel = network.apply(online_params, None, obs_t)
            q_sel = jnp.mean(jax.nn.softmax(q_logits_sel, axis=-1) * atoms, axis=-1)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)

            batch_error = jax.vmap(rlax.categorical_double_q_learning,
                                   in_axes=(None, 0, 0, 0, 0, None, 0, 0,))
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t, q_sel)
            return td_errors
        
        # calculate q td loss for distributional network
        def categorical_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            
            q_logits_tm1 = network.apply(online_params, None, obs_tm1)
            q_logits_t = network.apply(trg_params, None, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.categorical_q_learning,
                                   in_axes=(None, 0, 0, 0, 0, None, 0,))
            
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t)
            return td_errors
        
        # calculate double q td loss (no distributional output)
        def double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            
            q_tm1 = network.apply(online_params, None, obs_tm1)
            q_t = network.apply(trg_params, None, obs_t)
            q_t_selector = network.apply(online_params, None, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.double_q_learning,
                                   in_axes=(0, 0, 0, 0, 0, 0))
            
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t, q_t_selector)
            return td_errors
        
         # calculate q td loss (no distributional output)
        def q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            
            q_tm1 = network.apply(online_params, None, obs_tm1)
            q_t = network.apply(trg_params, None, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.q_learning,
                                   in_axes=(0, 0, 0, 0, 0,))
            
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t)
            return td_errors

        def loss(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t, prios):
            
            # importance sampling
            weights_is = (1. / prios).astype(jnp.float32) ** beta_is
            weights_is /= jnp.max(weights_is)

            # select the loss calculation function (either for double q learning or q learning)
            if use_distribution:
                q_loss_td = categorical_double_q_td if use_double_q else categorical_q_td
            else:
                q_loss_td = double_q_td if use_double_q else q_td
            batch_loss = q_loss_td(
                online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t
            )

            # importance sampling
            mean_loss = jnp.mean(batch_loss * weights_is)
            new_prios = jnp.abs(batch_loss)
            return mean_loss, new_prios


        grad_fn = jax.grad(loss, has_aux=True)
        grads, new_prios = grad_fn(
            online_params, trg_params,
            transitions["observation_tm1"],
            transitions["action_tm1"][:, 0],
            transitions["reward_t"][:, 0],
            transitions["observation_t"],
            transitions["terminal_t"],
            discount_t,
            prios
        )

        updates, opt_state_t = optimizer.update(grads, opt_state)
        online_params_t = optax.apply_updates(online_params, updates)
        return online_params_t, opt_state_t, new_prios


class DQNAgent:
    def __init__(
            self,
            observation_spec,
            action_spec,
            params: RlaxRainbowParams = RlaxRainbowParams(),
            reward_shaper = None):

        if not callable(params.epsilon):
            eps = params.epsilon
            params = params._replace(epsilon=lambda ts: eps)
        if not callable(params.beta_is):
            beta = params.beta_is
            params = params._replace(beta_is=lambda ts: beta)

        self.params = params
        self.reward_shaper = reward_shaper
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(params.seed))
        self.drawn_priorities = []
        self.n_network = params.n_network

        # Function to build and initialize Q-network. Use Noisy Network or MLP
        def build_network(
                layers: List[int],
                output_shape: List[int],
                use_noisy_network: bool) -> hk.Transformed:

            def q_net(obs):
                layers_ = tuple(layers) + (onp.prod(output_shape), )
                if use_noisy_network:
                    network = NoisyMLP(layers_, factorized_noise=self.params.factorized_noise) 
                else:
                    network = hk.nets.MLP(layers_)
                return hk.Reshape(output_shape=output_shape)(network(obs))

            return hk.transform(q_net)
                
        # define shape of output layer
        if self.params.use_distribution:
            output_shape = (action_spec.num_values, params.n_atoms)
        else:
            output_shape = (action_spec.num_values,)

        # create the network
        self.network = build_network(
            params.layers, # layers
            output_shape, # output shape
            params.use_noisy_network # network type
        )
        
        # input vector of wanted shape filled with zeros
        input_placeholder = onp.zeros(
            (observation_spec.shape[0], observation_spec.shape[1] * self.params.history_size),
            dtype = onp.float16
        )
        
        # keys list
        keys = jnp.array([next(self.rng) for _ in range(self.n_network)])
        # initialize networks
        parallel_network_init = jax.vmap(self.network.init, in_axes=(0, None))
        self.trg_params = parallel_network_init(keys, input_placeholder)
        
        self.online_params = self.trg_params
        
        # create atoms for distributional network
        self.atoms = jnp.tile(
            jnp.linspace(-params.atom_vmax, params.atom_vmax, params.n_atoms), 
            (action_spec.num_values, 1)
        )

        # Build and initialize Adam optimizer.
        self.optimizer = optax.adam(params.learning_rate, eps=3.125e-5)
        parallel_opt_init = jax.vmap(self.optimizer.init, in_axes=(0,))
        self.opt_state = parallel_opt_init(self.online_params)
        
        #self.opt_state = self.optimizer.init(self.online_params)
        self.train_step = 0
        
        # Define the update function
        self.update_q = DQNLearning.update_q

        # Create Buffer (Priority Buffer or Experience Buffer)
        if params.use_priority:
            self.experience = [
                PriorityBuffer(observation_spec.shape[1] * self.params.history_size, 
                               action_spec.num_values, 
                               1,
                               params.experience_buffer_size,
                               alpha=self.params.priority_w) 
                for _ in range(self.n_network)
            ]
        else:
            self.experience = [
                ExperienceBuffer(observation_spec.shape[1] * self.params.history_size,
                                 action_spec.num_values,
                                 1,
                                 params.experience_buffer_size)
                for _ in range(self.n_network)
            ]
           
        self.requires_vectorized_observation = lambda: True

    def exploit(self, observations):
        observations, legal_actions = observations[1]
        
        obs = observations.reshape(self.n_network, -1, observations.shape[1])
        vla = legal_actions.reshape(self.n_network, -1, legal_actions.shape[1])
        keys = jnp.array([next(self.rng) for _ in range(self.n_network)])
        
        parallel_eval = jax.vmap(DQNPolicy.eval_policy, in_axes=(None, None, None, 0, 0, 0, 0))
        actions = parallel_eval(self.network, self.params.use_distribution, self.atoms, 
                                self.online_params, keys, obs, vla)
        
        return jax.tree_util.tree_map(onp.array, actions).flatten()

    def explore(self, observations):
        observations, legal_actions = observations[1]
        
        obs = observations.reshape(self.n_network, -1, observations.shape[1])
        vla = legal_actions.reshape(self.n_network, -1, legal_actions.shape[1])
        keys = jnp.array([next(self.rng) for _ in range(self.n_network)])
        
        parallel_eval = jax.vmap(DQNPolicy.policy, in_axes=(None, None, None, 0, None, 0, 0, 0))
        _, actions = parallel_eval(self.network, self.params.use_distribution, self.atoms,
                                self.online_params, self.params.epsilon(self.train_step), keys, obs, vla)
        
        return jax.tree_util.tree_map(onp.array, actions).flatten()
    
    def add_experience_first(self, observations, step_types):
        pass

    def add_experience(self, observations_tm1, actions_tm1, rewards_t, observations_t, term_t):

        obs_vec_tm1 = observations_tm1[1][0]
        obs_vec_t = observations_t[1][0]
        legal_actions_t = observations_t[1][1]
        
        # reshape input
        obs_vec_tm1 = obs_vec_tm1.reshape(self.n_network, -1 , obs_vec_tm1.shape[1])
        obs_vec_t = obs_vec_t.reshape(self.n_network, -1 , obs_vec_t.shape[1])
        legal_actions_t = legal_actions_t.reshape(self.n_network, -1 , legal_actions_t.shape[1])
        actions_tm1 = actions_tm1.reshape(self.n_network, -1, 1)
        rewards_t = rewards_t.reshape(self.n_network, -1, 1)
        term_t = term_t.reshape(self.n_network, -1, 1)
        
        for i in range(self.n_network):
            self.experience[i].add_transitions(
                obs_vec_tm1[i],
                actions_tm1[i],
                rewards_t[i],
                obs_vec_t[i],
                legal_actions_t[i],
                term_t[i]
            )
        
    def shape_rewards(self, observations, moves):
        
        if self.reward_shaper is not None:
            shaped_rewards, shape_type = self.reward_shaper.shape(observations[0], 
                                                                  moves,
                                                                  self.train_step)
            return onp.array(shaped_rewards), onp.array(shape_type)
        return (onp.zeros(len(observations[0])), onp.zeros(len(observations[0])))

    def update(self):
        """Make one training step.
        """
        
        def sample(experience):
            if self.params.use_priority:
                sample_indices, prios, transitions = experience.sample_batch(self.params.train_batch_size)
                return sample_indices, prios, transitions
            else:
                transitions = experience.sample(self.params.train_batch_size)
                prios = onp.ones(transitions["observation_tm1"].shape[0])
                return None, prios, transitions
            
        def combine_dict(lst_dict):
            dict_out = {}
            for keys in lst_dict[0]:
                dict_out[keys] = onp.array([d[keys] for d in lst_dict])
            return dict_out

        if not self.params.fixed_weights:

            sample_indices, prios, transitions = zip(*map(sample, self.experience))
            transitions = combine_dict(transitions)
                
            parallel_update = jax.vmap(self.update_q, in_axes=(None, None, None, 0, 0, 0, 
                                                               0, None, 0, None, None, None))
            
            self.online_params, self.opt_state, tds = parallel_update(
                self.network,
                self.atoms,
                self.optimizer,
                self.online_params,
                self.trg_params,
                self.opt_state,
                transitions,
                self.params.discount,
                onp.array(prios),
                self.params.beta_is(self.train_step),
                self.params.use_double_q,
                self.params.use_distribution)
            
            if self.params.use_priority:
                for i in range(self.n_network):
                    self.experience[i].update_priorities(sample_indices[i], onp.abs(tds[i]))
    
            if self.train_step % self.params.target_update_period == 0:
                self.trg_params = self.online_params
    
            self.train_step += 1
        
    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.params.history_size, 
                                            obs_len,
                                            n_states)

    def __repr__(self):
        return f"<rlax_dqn.DQNAgent(params={self.params})>"
    
    def save_weights(self, path, fname_part, only_weights=True):
        """Save online and target network weights to the specified path
        added: save optimizer state"""
        
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_online.pkl"), 'wb') as of:
            pickle.dump(self.online_params, of)
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_target.pkl"), 'wb') as of:
            pickle.dump(self.trg_params, of)
            
        if not only_weights:
            with open(join_path(path, "rlax_rainbow_" + fname_part + "_opt_state.pkl"), 'wb') as of:
                pickle.dump(jax.tree_util.tree_map(onp.array, self.opt_state), of)
            with open(join_path(path, "rlax_rainbow_" + fname_part + "_experience.pkl"), 'wb') as of:
                pickle.dump([e.serializable() for e in self.experience], of)
    
    # older versions of haiku store weights as frozendict, convert to mutable dict and then to FlatMapping
    def _compat_restore_weights(self, file_w):
        weights = pickle.load(file_w)
        mutable = hk.data_structures.to_mutable_dict(weights)
        for m in mutable:
            mutable[m] = hk.data_structures.to_mutable_dict(mutable[m])
        return hk.data_structures.to_immutable_dict(mutable)

    def restore_weights(self, online_weights_file, 
                        trg_weights_file, 
                        opt_state_file=None, 
                        experience_file=None):
        """Restore online and target network weights from the specified files
        added: load optimizer state if file name given"""
        with open(online_weights_file, 'rb') as iwf:
            self.online_params = self._compat_restore_weights(iwf)#pickle.load(iwf)
        with open(trg_weights_file, 'rb') as iwf:
            self.trg_params = self._compat_restore_weights(iwf)#pickle.load(iwf)
        # optimizer state
        if opt_state_file is not None:
            with open(opt_state_file, 'rb') as iwf:
                self.opt_state = pickle.load(iwf)
            self.train_step = onp.asscalar(self.opt_state[0].count)
        # experience buffer
        if experience_file is not None:
            with open(experience_file, 'rb') as iwf:
                for i, serialized in enumerate(pickle.load(iwf)):
                    self.experience[i].load(serialized)
                
#     def get_buffer_priorities(self):
#         if self.params.use_priority:
#             index_list = range(self.experience.size)
#             return self.experience.get_priorities(index_list)
#         else:
#             return None
#         
#     def get_drawn_priorities(self, reset=True):
#         prios = self.drawn_priorities.copy()
#         self.drawn_priorities.clear()
#         return prios
#     
#     def get_stochasticity(self):
#         
#         def process_weights(w):
#             return onp.mean(onp.abs(w['w_sigma']))
#         
#         if self.params.use_noisy_network:
#             weights = hk.data_structures.to_mutable_dict(self.online_params)
#             return [process_weights(w) for w in weights.values()]
#         else:
#             return None

