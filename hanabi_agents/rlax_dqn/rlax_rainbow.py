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
    
    def _apply_legal_boltzmann(probs, tau, legal):
        """Mix an arbitrary categorical distribution with a boltzmann distribution"""
        boltzmann_probs = jnp.where(legal, jnp.exp(probs / tau), 0.)
        boltzmann_probs = boltzmann_probs / jnp.sum(boltzmann_probs, axis=-1)[:, None]
        return boltzmann_probs

    @staticmethod
    def _argmax_with_random_tie_breaking(preferences):
        """Compute probabilities greedily with respect to a set of preferences."""
        optimal_actions = (preferences == preferences.max(axis=-1, keepdims=True))
        return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)
    
    @staticmethod
    def legal_softmax(tau=None):
        """An epsilon-greedy distribution with illegal probabilities set to zero"""

        def sample_fn(key: chex.Array,
                      preferences: chex.Array,
                      legal: chex.Array,
                      tau=tau):
            probs = DQNPolicy._apply_legal_boltzmann(probs, tau, legal)
            return DQNPolicy._categorical_sample(key, probs)

        def probs_fn(preferences: chex.Array, legal: chex.Array, tau=tau):
            return DQNPolicy._apply_legal_boltzmann(probs, tau, legal)

        def logprob_fn(sample: chex.Array,
                       preferences: chex.Array,
                       legal: chex.Array,
                       tau=tau):
            probs = DQNPolicy._apply_legal_boltzmann(probs, tau, legal)
            return rlax.base.batched_index(jnp.log(probs), sample)

        def entropy_fn(preferences: chex.Array, legal: chex.Array, epsilon=epsilon):
            probs = DQNPolicy._apply_legal_boltzmann(probs, tau, legal)
            return -jnp.nansum(probs * jnp.log(probs), axis=-1)

        return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)

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
    @partial(jax.jit, static_argnums=(0,1,2))
    def policy(
            network,
            use_distribution: bool,
            use_softmax: bool,
            atoms,
            net_params,
            epsilon: float,
            tau: float,
            key: float,
            obs: chex.Array,
            lms: chex.Array):
        """Sample action from epsilon-greedy policy.

        Args:
            network            -- haiku Transformed network.
            use_distribution   -- network has distributional output
            atoms              -- support of distributional output
            net_params         -- parameters (weights) of the network.
            epsilon            -- proportion of uniform sampling for epsilon greedy sampling
            tau                -- annealing temperature for softmax sampling
            key                -- key for categorical sampling.
            obs                -- observation.
            lm                 -- one-hot encoded legal actions
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
        if use_softmax:
            actions = DQNPolicy.legal_softmax(tau=tau).sample(key, q_vals, lms)
        else:
            actions = DQNPolicy.legal_epsilon_greedy(epsilon=epsilon).sample(key, q_vals, lms)
        return q_vals, actions

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def eval_policy(
            network,
            use_distribution,
            atoms,
            net_params,
            key,
            obs: chex.Array,
            lms: chex.Array):
        """Sample action from greedy policy.
        Args:
            network            -- haiku Transformed network.
            use_distribution   -- network has distributional output
            atoms              -- support of distributional output
            net_params         -- parameters (weights) of the network.
            key                -- key for categorical sampling.
            obs                -- observation.
            lm                 -- one-hot encoded legal actions
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
    @partial(jax.jit, static_argnums=(0, 1, 2, 3))
    def update_q(network, optimizer, use_distribution, use_double_q, # static arguments
                 atoms, online_params, trg_params, opt_state, transitions, discount_t, prios, beta_is):
        """Update network weights wrt Q-learning loss.

        Args:
            network            -- haiku Transformed network.
            optimizer          -- optimizer
            use_distribution   -- network has distributional output
            use_double_q       -- use double q learning
            atoms              -- support of distributional output
            online params      -- weights of the online network
            trg_params         -- weights of the target network
            opt_state          -- state of the optimizer.
            transitions        -- batch of transitions(obs_tm1, act_tm1, reward_t, obs_t, terminal_t)
            discount_t         -- discount factor
            beta_is            -- importance sampling exponent
        """


        def categorical_double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            """
            calculate double q td loss for distributional network
            """
            q_logits_tm1 = network.apply(online_params, None, obs_tm1)
            q_logits_t = network.apply(trg_params, None, obs_t)
            q_logits_sel = network.apply(online_params, None, obs_t)
            q_sel = jnp.mean(jax.nn.softmax(q_logits_sel, axis=-1) * atoms, axis=-1)
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            batch_error = jax.vmap(rlax.categorical_double_q_learning, in_axes=(None, 0, 0, 0, 0, None, 0, 0,))
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t, q_sel)
            return td_errors
        
        def categorical_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            """
            calculate q td loss for distributional network
            """
            q_logits_tm1 = network.apply(online_params, None, obs_tm1)
            q_logits_t = network.apply(trg_params, None, obs_t)
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            batch_error = jax.vmap(rlax.categorical_q_learning, in_axes=(None, 0, 0, 0, 0, None, 0,))
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t)
            return td_errors
        
        def double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            """
            calculate double q td loss (no distributional output)
            """
            q_tm1 = network.apply(online_params, None, obs_tm1)
            q_t = network.apply(trg_params, None, obs_t)
            q_t_selector = network.apply(online_params, None, obs_t)
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            batch_error = jax.vmap(rlax.double_q_learning, in_axes=(0, 0, 0, 0, 0, 0))
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t, q_t_selector)
            return td_errors
        
        def q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            """
            calculate q td loss (no distributional output)
            """
            q_tm1 = network.apply(online_params, None, obs_tm1)
            q_t = network.apply(trg_params, None, obs_t)
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            batch_error = jax.vmap(rlax.q_learning, in_axes=(0, 0, 0, 0, 0,))
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t)
            return td_errors

        def loss(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t, prios):
            weights_is = (1. / prios).astype(jnp.float32) ** beta_is
            weights_is /= jnp.max(weights_is)
            
            if use_distribution:
                loss_fct = categorical_double_q_td if use_double_q else categorical_q_td
            else:
                loss_fct = double_q_td if use_double_q else q_td

            batch_loss = loss_fct(
                online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t
            )

            mean_loss = jnp.mean(batch_loss * weights_is)
            new_prios = jnp.abs(batch_loss)
            print(new_prios)
            return mean_loss, new_prios


        grad_fn = jax.grad(loss, has_aux=True)
        grads, new_prios = grad_fn(
            online_params, 
            trg_params,
            transitions.observation_tm1,
            transitions.action_tm1[:, 0],
            transitions.reward_t[:, 0],
            transitions.observation_t,
            transitions.terminal_t,
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
        if not callable(params.tau):
            tau = params.tau
            params = params._replace(tau=lambda ts: tau)
        if not callable(params.beta_is):
            beta = params.beta_is
            params = params._replace(beta_is=lambda ts: beta)
        self.params = params
        self.reward_shaper = reward_shaper
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(params.seed))
        
        # function to create network structure
        def build_network(
                layers: List[int],
                output_shape: List[int],
                use_noisy_network: bool,
                use_factorized_noise: bool
            ) -> hk.Transformed:

            def q_net(obs):
                layers_ = tuple(layers) + (onp.prod(output_shape), )
                if use_noisy_network:
                    network = NoisyMLP(layers_, factorized_noise=use_factorized_noise)
                else:
                    network = hk.nets.MLP(layers_)
                return hk.Reshape(output_shape=output_shape)(network(obs))

            return hk.transform(q_net)
        
        # define shape of output layer
        if self.params.use_distribution:
            output_shape = (action_spec.num_values, params.n_atoms)
        else:
            output_shape = (action_spec.num_values,)
        
        # define input placeholder needed for initialization of network
        input_placeholder = onp.zeros(
            (observation_spec.shape[0], observation_spec.shape[1] * self.params.history_size),
            dtype = onp.float16
        )
            
        # create network structure
        self.network = build_network(
            self.params.layers, # number of neurons in hidden layers
            output_shape, # shape of output layer
            params.use_noisy_network, # layer type
            params.factorized_noise # (de)activate factorized noise
        )
        
        # initialize network
        self.trg_params = self.network.init(next(self.rng), input_placeholder)
        self.online_params = self.trg_params
        
        # initialize support for distributional output (for each action)
        support = jnp.linspace(-params.atom_vmax, params.atom_vmax, params.n_atoms)
        self.atoms = jnp.tile(support, (action_spec.num_values, 1))

        # Build and initialize optimizer.
        self.optimizer = optax.adam(params.learning_rate, eps=3.125e-5)
        self.opt_state = self.optimizer.init(self.online_params)
        
        # Define update function (for training)
        self.update_q = DQNLearning.update_q

        # Initialize experience buffer
        if params.use_priority:
            self.experience = PriorityBuffer(
                observation_spec.shape[1] * self.params.history_size, # shape of observation
                params.experience_buffer_size, # buffer size
                alpha=self.params.priority_w # prioritization exponent
            ) 
        else:
            self.experience = ExperienceBuffer(
                observation_spec.shape[1] * self.params.history_size, # shape of observation
                params.experience_buffer_size # buffer size
            )

        self.requires_vectorized_observation = lambda: True
        
        self.train_step = 0

    def exploit(self, observations):
        """
        Exploitation, greedy selection of actions.
        """
        observations, legal_actions = observations[1]
        actions = DQNPolicy.eval_policy(
            self.network, self.params.use_distribution, 
            self.atoms, self.online_params,
            next(self.rng), observations, legal_actions
        )
        return jax.tree_util.tree_map(onp.array, actions)

    def explore(self, observations):
        """
        Exploration, epsilon-greedy selection of actions.
        """
        observations, legal_actions = observations[1]
        _, actions = DQNPolicy.policy(
            self.network, self.params.use_distribution, self.params.use_boltzmann_exploration,
            self.atoms, self.online_params,
            self.params.epsilon(self.train_step), self.params.tau(self.train_step),
            next(self.rng), observations, legal_actions
        )
        return jax.tree_util.tree_map(onp.array, actions)
    
    def add_experience_first(self, observations, step_types):
        pass

    def add_experience(self, observations_tm1, actions_tm1, rewards_t, observations_t, term_t):
        """
        Add transition batch to experience buffer
        """

        obs_vec_tm1 = observations_tm1[1][0]
        obs_vec_t = observations_t[1][0]

        self.experience.add_transitions(
            obs_vec_tm1,
            actions_tm1.reshape(-1,1),
            rewards_t,
            obs_vec_t,
            term_t
        )
        
    def shape_rewards(self, observations, moves):
        """
        Apply reward shaping function to list of HanabiObservation, HanabiMove
        if shaper object is defined at initialization
        """
        if self.reward_shaper is not None:
            shaped_rewards, shape_type = self.reward_shaper.shape(observations[0], 
                                                                  moves,
                                                                  self.train_step)
            return onp.array(shaped_rewards), onp.array(shape_type)
        return (onp.zeros(len(observations[0])), onp.zeros(len(observations[0])))

    def update(self):
        """Make one training step.
        """
        
        if not self.params.fixed_weights:

            # sample batch of transitions from buffer
            if self.params.use_priority:
                sample_indices, prios, transitions = self.experience.sample_batch(
                    self.params.train_batch_size)
            else:
                transitions = self.experience.sample_batch(self.params.train_batch_size)
                prios = onp.ones(transitions.observation_tm1.shape[0])
    
            # call update function, calculate td error and network state
            self.online_params, self.opt_state, tds = self.update_q(
                self.network,
                self.optimizer,
                self.params.use_distribution,
                self.params.use_double_q,
                self.atoms,
                self.online_params,
                self.trg_params,
                self.opt_state,
                transitions,
                self.params.discount,
                prios,
                self.params.beta_is(self.train_step))

            # update priorities in buffer
            if self.params.use_priority:
                tds = jax.tree_util.tree_map(onp.array, tds)
                self.experience.update_priorities(sample_indices, tds)
    
            # periodically update target network
            if self.train_step % self.params.target_update_period == 0:
                self.trg_params = self.online_params
    
            self.train_step += 1
        
    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.params.history_size, obs_len, n_states)

    def __repr__(self):
        return f"<rlax_dqn.DQNAgent(params={self.params})>"
      
    def save_weights(self, path, fname_part, only_weights=True):
        """
        Save online and target network weights to the specified path
        added: optionally save optimizer state and experience buffer
        """
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_online.pkl"), 'wb') as of:
            pickle.dump(self.online_params, of)
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_target.pkl"), 'wb') as of:
            pickle.dump(self.trg_params, of)
        if not only_weights:
            with open(join_path(path, "rlax_rainbow_" + fname_part + "_opt_state.pkl"), 'wb') as of:
                pickle.dump(jax.tree_util.tree_map(onp.array, self.opt_state), of)
            with open(join_path(path, "rlax_rainbow_" + fname_part + "_experience.pkl"), 'wb') as of:
                pickle.dump(self.experience.serializable(), of)
    
    def _compat_restore_weights(self, file_w):
        """
        older versions of haiku store weights as frozendict, 
        for compatibility convert to mutable dict and then to FlatMapping
        """
        weights = pickle.load(file_w)
        mutable = hk.data_structures.to_mutable_dict(weights)
        for m in mutable:
            mutable[m] = hk.data_structures.to_mutable_dict(mutable[m])
        return hk.data_structures.to_immutable_dict(mutable)

    def restore_weights(self, online_weights_file, 
                        trg_weights_file=None, 
                        opt_state_file=None, 
                        experience_file=None):
        """
        Restore online and target network weights from the specified files
        added: if only online network is given: use online weights also for target network
        added: load optimizer state and experience buffer if file names are given
        """
        with open(online_weights_file, 'rb') as iwf:
            self.online_params = self._compat_restore_weights(iwf)
            
        # weights of target network
        if trg_weights_file is not None:
            with open(trg_weights_file, 'rb') as iwf:
                self.trg_params = self._compat_restore_weights(iwf)
        else:
            self.trg_params = self.online_params
            
        # optimizer state
        if opt_state_file is not None:
            with open(opt_state_file, 'rb') as iwf:
                self.opt_state = pickle.load(iwf)
            self.train_step = onp.asscalar(self.opt_state[0].count)

        # experience buffer
        if experience_file is not None:
            with open(experience_file, 'rb') as iwf:
                self.experience.load(pickle.load(iwf))
        
