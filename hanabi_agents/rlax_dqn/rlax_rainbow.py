"""
This file implements a DQNAgent.
"""
import collections
import pickle
from functools import partial
from typing import Tuple, List
from os.path import join as join_path

import numpy as onp
import random

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import rlax
import chex
import ray

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
    @partial(jax.jit, static_argnums=(0))
    def policy(
            network,
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
        # compute logits
        logits = network.apply(net_params, None, obs)
        # set logits for illegal actions to negative infinity
        #  tiled_lms = jnp.broadcast_to(lms[:, :, onp.newaxis], logits.shape)
        #  logits = jnp.where(tiled_lms, logits, -jnp.inf)
        probs = jax.nn.softmax(logits, axis=-1)
        q_vals = jnp.mean(probs * atoms, axis=-1)
        q_vals = jnp.where(lms, q_vals, -jnp.inf)

        # compute actions
        actions = DQNPolicy.legal_epsilon_greedy(epsilon=epsilon).sample(key, q_vals, lms)
        return q_vals, actions

    @staticmethod
    @partial(jax.jit, static_argnums=(0))
    def eval_policy(
            network,
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
        # compute logits and convert those to q_vals
        logits = network.apply(net_params, None, obs)
        probs = jax.nn.softmax(logits, axis=-1)
        q_vals = jnp.mean(probs * atoms, axis=-1)
        q_vals = jnp.where(lms, q_vals, -jnp.inf)

        # compute actions
        return rlax.greedy().sample(key, q_vals)

class DQNLearning:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2))
    def update_q(network, atoms, optimizer, online_params, trg_params, opt_state,
                 transitions, discount_t, prios, beta_is):
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

        def categorical_double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t):
            q_logits_tm1 = network.apply(online_params, None, obs_tm1)
            q_logits_t = network.apply(trg_params, None, obs_t)
            q_logits_sel = network.apply(online_params, None, obs_t)
            q_sel = jnp.mean(jax.nn.softmax(q_logits_sel, axis=-1) * atoms, axis=-1)
            # set q values of illegal actions to a large negative number.
            #  q_sel = jnp.where(lm_t, q_sel, -1e2)
            #  q_t = jnp.where(lm_t, q_t, -1e3)
            # set q values to zero if the state is terminal, i.e.
            #  q_t = jnp.where(jnp.broadcast_to(term_t, q_t.shape), 0.0, q_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)

            batch_error = jax.vmap(rlax.categorical_double_q_learning,
                                   in_axes=(None, 0, 0, 0, 0, None, 0, 0,))
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t, q_sel)
            return td_errors

        def loss(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t, prios):
            weights_is = (1. / prios).astype(jnp.float32) ** beta_is
            weights_is /= jnp.max(weights_is)

            batch_loss = categorical_double_q_td(
                online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t,
                term_t, discount_t)

            #  batch_loss = rlax.huber_loss(td_errors, 1.)
            mean_loss = jnp.mean(batch_loss * weights_is)

            new_prios = jnp.abs(batch_loss)

            return mean_loss, new_prios


        grad_fn = jax.grad(loss, has_aux=True)
        grads, new_prios = grad_fn(
            online_params, trg_params,
            transitions.observation_tm1,
            transitions.action_tm1[:, 0],
            transitions.reward_t[:, 0],
            transitions.observation_t,
            transitions.legal_moves_t,
            transitions.terminal_t,
            discount_t,
            prios)

        updates, opt_state_t = optimizer.update(grads, opt_state)
        online_params_t = optax.apply_updates(online_params, updates)
        return online_params_t, opt_state_t, new_prios

@ray.remote(num_gpus = 0.2)
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

        # Build and initialize Q-network.
        def build_network(
                layers: List[int],
                output_shape: List[int]) -> hk.Transformed:

            def q_net(obs):
                layers_ = tuple(layers) + (onp.prod(output_shape), )
                network = NoisyMLP(layers_)
                return hk.Reshape(output_shape=output_shape)(network(obs))

            return hk.transform(q_net)

        self.network = build_network(params.layers,
                                     (action_spec.num_values, params.n_atoms))
        self.trg_params = self.network.init(
            next(self.rng), 
            onp.zeros((observation_spec.shape[0], observation_spec.shape[1] * self.params.history_size), dtype = onp.float16))
        self.online_params = self.trg_params
        self.atoms = jnp.tile(jnp.linspace(-params.atom_vmax, params.atom_vmax, params.n_atoms),
                              (action_spec.num_values, 1))

        # Build and initialize optimizer.
        self.optimizer = optax.adam(params.learning_rate, eps=3.125e-5)
        self.opt_state = self.optimizer.init(self.online_params)
        self.train_step = 0
        self.update_q = DQNLearning.update_q

        self.learning_rate = self.params.learning_rate
        self.buffersize = self.params.experience_buffer_size

        if params.use_priority:
            self.experience = PriorityBuffer(
                observation_spec.shape[1] * self.params.history_size,
                action_spec.num_values,
                1,
                params.experience_buffer_size,
                alpha=self.params.priority_w)
        else:
            self.experience = ExperienceBuffer(
                observation_spec.shape[1] * self.params.history_size,
                action_spec.num_values,
                1,
                params.experience_buffer_size)
        self.last_obs = onp.empty(observation_spec.shape)
        self.requires_vectorized_observation = lambda: True

    def exploit(self, observations):
        observations, legal_actions = observations[1]
        actions = DQNPolicy.eval_policy(
            self.network, self.atoms, self.online_params,
            next(self.rng), observations, legal_actions)
        return jax.tree_util.tree_map(onp.array, actions)

    def explore(self, observations):
        observations, legal_actions = observations[1]
        _, actions = DQNPolicy.policy(
            self.network, self.atoms, self.online_params,
            self.params.epsilon(self.train_step), next(self.rng),
            observations, legal_actions)
        return jax.tree_util.tree_map(onp.array, actions)
    
    # deprecated with new implementation for stacking
    def add_experience_first(self, observations, step_types):
        pass

    def add_experience(self, observations_tm1, actions_tm1, rewards_t, observations_t, term_t):

        obs_vec_tm1 = observations_tm1[1][0]
        obs_vec_t = observations_t[1][0]
        legal_actions_t = observations_t[1][1]

        self.experience.add_transitions(
            obs_vec_tm1,
            actions_tm1,
            rewards_t,
            obs_vec_t,
            legal_actions_t,
            term_t)
        
    def shape_rewards(self, observations, moves):
        
        if self.reward_shaper is not None:
            shaped_rewards, shape_type = self.reward_shaper.shape(observations[0], 
                                                                  moves)
            return onp.array(shaped_rewards), onp.array(shape_type)
        return (0, 0)

    def update(self):
        """Make one training step.
        """

        if self.params.use_priority:
            sample_indices, prios, transitions = self.experience.sample_batch(
                self.params.train_batch_size)
        else:
            transitions = self.experience.sample(self.params.train_batch_size)
            prios = onp.ones(transitions.observation_tm1.shape[0])

        self.online_params, self.opt_state, tds = self.update_q(
            self.network,
            self.atoms,
            self.optimizer,
            self.online_params,
            self.trg_params,
            self.opt_state,
            transitions,
            self.params.discount,
            prios,
            self.params.beta_is(self.train_step))

        if self.params.use_priority:
            self.experience.update_priorities(sample_indices, onp.abs(tds))

        if self.train_step % self.params.target_update_period == 0:
            self.trg_params = self.online_params

        self.train_step += 1
        
    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.params.history_size, 
                                            obs_len,
                                            n_states)

    def __repr__(self):
        return f"<rlax_dqn.DQNAgent(params={self.params})>"

    def save_weights(self, path, fname_part):
        """Save online and target network weights to the specified path"""

        # TODO save weights using something other than pickle (e.g. numpy + protobuf)
        #  flat_params, tree_def = jax.tree_util.tree_flatten(self.online_params)
        #  print(flat_params, tree_def)
        #  onp.save(join_path(path, "rlax_rainbow_" + fname_part + "_" + str(self.train_step) + "_online.npy"),
        #           self.online_params)
        #  onp.save(join_path(path, "rlax_rainbow_" + fname_part + "_" + str(self.train_step) + "_target.npy"),
        #           self.trg_params)
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_online.pkl"), 'wb') as of:
            pickle.dump(self.online_params, of)
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_target.pkl"), 'wb') as of:
            pickle.dump(self.trg_params, of)

    def restore_weights(self, online_weights_file, trg_weights_file):
        """Restore online and target network weights from the specified files"""

        with open(online_weights_file, 'rb') as iwf:
            self.online_params = pickle.load(iwf)
        with open(trg_weights_file, 'rb') as iwf:
            self.trg_params = pickle.load(iwf)

    """Functions necessary for PBT"""

    def get_agent_attributes(self):
        """Retrieves network weights, to copy them over to other models"""
        # network_weights = self.online_params
        attributes = (self.learning_rate, self.buffersize, self.online_params)
        return attributes

    def overwrite_weights(self, network_weights):
        """Overwrites this model's network weights with those taken from another model in the Population"""
        self.online_params = network_weights
        self.trg_params = network_weights

    def overwrite_lr(self, lr_factor, lr_survivor):
        """Resets the optimizer with altered learning rate"""
        self.learning_rate = lr_survivor * (1 + (random.randint(0,1) * 2 - 1) * lr_factor)
        # self.params._replace(learning_rate = new_lr)
        self.optimizer = optax.adam(self.learning_rate, eps=3.125e-5)
        self.opt_state = self.optimizer.init(self.online_params)

    def change_buffersize(self, buffer_factor, buffersize_survivor):
        """Alters the size of the current model's buffer_size randomly by buffer_factor up/down"""
        if self.buffersize <= 512:
            choices = [buffer_factor, 1]
        elif self.buffersize >= 2**15:
            choices = [1, 1 / buffer_factor]
        else:
            choices = [buffer_factor, 1/buffer_factor]
        self.buffersize = int(buffersize_survivor * random.choice(choices))

        self.experience.change_size(self.buffersize)

        #TODO: implement for priority replay