"""
This file implements a DQNAgent.
"""
import collections
from functools import partial
from typing import Sequence, Tuple, List

import numpy as onp

import haiku as hk
from haiku import nets
import jax
from jax.experimental import optix
import jax.numpy as jnp
import rlax

from .experience_buffer import ExperienceBuffer
from .priority_buffer import PriorityBuffer
from .transition import Transition


DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy"])


def build_network(layers: list, num_actions: int) -> hk.Transformed:

    def q_net(obs):
        layers_ = tuple(layers) + (num_actions, )
        network = hk.nets.MLP(layers_)
        return network(obs)

    return hk.transform(q_net)


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

        def sample_fn(key: rlax.ArrayLike,
                      preferences: rlax.ArrayLike,
                      legal: rlax.ArrayLike,
                      epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return DQNPolicy._categorical_sample(key, probs)

        def probs_fn(preferences: rlax.ArrayLike, legal: rlax.ArrayLike, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            return DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)

        def logprob_fn(sample: rlax.ArrayLike,
                       preferences: rlax.ArrayLike,
                       legal: rlax.ArrayLike,
                       epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return rlax.base.batched_index(jnp.log(probs), sample)

        def entropy_fn(preferences: rlax.ArrayLike, legal: rlax.ArrayLike, epsilon=epsilon):
            probs = DQNPolicy._argmax_with_random_tie_breaking(preferences)
            probs = DQNPolicy._mix_with_legal_uniform(probs, epsilon, legal)
            return -jnp.nansum(probs * jnp.log(probs), axis=-1)

        return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def policy(network,
               net_params,
               epsilon: float,
               key: float,
               obs: rlax.ArrayLike,
               lm: rlax.ArrayLike):
        """Sample action from epsilon-greedy policy.

        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q
        q_vals = network.apply(net_params, obs)
        # set q for illegal actions to negative infinity
        q_vals = jnp.where(lm, q_vals, -jnp.inf)
        # compute actions
        actions = DQNPolicy.legal_epsilon_greedy(epsilon=epsilon).sample(key, q_vals, lm)
        return q_vals, actions

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def eval_policy(network, net_params, key, obs: rlax.ArrayLike, lm: rlax.ArrayLike):
        """Sample action from greedy policy.
        Args:
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        # compute q
        q_vals = network.apply(net_params, obs)
        # add large negative values to illegal actions
        q_vals = jnp.where(lm, q_vals, -jnp.inf)
        # compute actions
        return rlax.greedy().sample(key, q_vals)

class DQNLearning:
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def update_q(   network, 
                    optimizer, 
                    online_params, 
                    trg_params, 
                    opt_state,
                    transitions, 
                    discount_t, 
                    weights_is, 
                    importance_beta):
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

        def q_learning_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t):
            q_tm1 = network.apply(online_params, obs_tm1)
            q_t = network.apply(trg_params, obs_t)
            # set q values of illegal actions to a large negative number.
            #  q_t = jnp.where(lm_t, q_t, -1e3)
            # set q values to zero if the state is terminal, i.e.
            q_t = jnp.where(jnp.broadcast_to(term_t, q_t.shape), 0.0, q_t)
            return rlax.q_learning(q_tm1, a_tm1, r_t, discount_t, q_t)

        def double_q_learning_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t):
            q_tm1 = network.apply(online_params, obs_tm1)
            q_t = network.apply(trg_params, obs_t)
            q_sel = network.apply(online_params, obs_t)
            # set q values of illegal actions to a large negative number.
            #  q_sel = jnp.where(lm_t, q_sel, -1e2)
            #  q_t = jnp.where(lm_t, q_t, -1e3)
            # set q values to zero if the state is terminal, i.e.
            q_t = jnp.where(jnp.broadcast_to(term_t, q_t.shape), 0.0, q_t)
            td_errors = jax.vmap(rlax.double_q_learning, in_axes=(0, 0, 0, None, 0, 0,))(
                    q_tm1, a_tm1, r_t, discount_t, q_t, q_sel)
            return td_errors
            #  return rlax.l2_loss(td_errors).mean()

        def loss(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t, weights_is):
            #  idxes = self._sample_proportional(batch_size)
            #  weights = []
            #  p_min = self._it_min.min() / self._it_sum.sum()
            #  max_weight = (p_min * len(self._storage)) ** (-beta)
            #  p_sample = self._it_sum[idxes] / self._it_sum.sum()
            #  weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
            #  weights_is = jnp.power(
            #      priorities * transitions.observation_tm1.shape[0],
            #      -importance_beta)
            #  weights_is = weights_is * priorities
            #  td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
            #  errors = tf_util.huber_loss(td_error)
            #  weighted_error = tf.reduce_mean(importance_weights_ph * errors)
            #  gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
            return rlax.clip_gradient(
                jnp.mean(weights_is *
                rlax.l2_loss(
                    double_q_learning_td(
                        online_params, trg_params,
                        obs_tm1, a_tm1, r_t, obs_t, lm_t, term_t, discount_t))),
                -1, 1)


        #  tds = jax.vmap(double_q_learning_td, in_axes=(None, None, 0, 0, 0, 0, 0, 0, None))(
        #          online_params, trg_params, transitions.observation_tm1, transitions.action_tm1[:, 0], transitions.reward_t[:, 0], transitions.observation_t, transitions.legal_moves_t, transitions.terminal_t, discount_t)

        td_errors = double_q_learning_td(   online_params, trg_params,
                                            transitions.observation_tm1,
                                            transitions.action_tm1[:, 0],
                                            transitions.reward_t[:, 0],
                                            transitions.observation_t,
                                            transitions.legal_moves_t,
                                            transitions.terminal_t,
                                            discount_t)

        dloss_dtheta = jax.grad(loss)(  online_params, trg_params,
                                        transitions.observation_tm1,
                                        transitions.action_tm1[:, 0],
                                        transitions.reward_t[:, 0],
                                        transitions.observation_t,
                                        transitions.legal_moves_t,
                                        transitions.terminal_t,
                                        discount_t,
                                        weights_is)
        #  print(dloss_dtheta)
        #  dloss_dtheta = jax.grad(loss)(td_errors)
        updates, opt_state_t = optimizer.update(dloss_dtheta, opt_state)
        online_params_t = optix.apply_updates(online_params, updates)
        return online_params_t, opt_state_t, td_errors

class DQNAgent:
    def __init__(
            self,
            observation_spec,
            action_spec,
            target_update_period: int = None,
            discount: float = None,
            epsilon=lambda x: 0.1,
            learning_rate: float = 0.001,
            layers: List[int] = None,
            use_double_q=True,
            use_priority=True,
            seed: int = 1234,
            importance_beta: float = 0.4):
        self.rng = hk.PRNGSequence(jax.random.PRNGKey(seed))

        # Build and initialize Q-network.
        self.layers = layers or (512,)
        self.network = build_network(self.layers, action_spec.num_values)
        #  sample_input = env.observation_spec()["observation"].generate_value()
        #  sample_input = jnp.zeros(observation_le)
        self.trg_params = self.network.init(next(self.rng), observation_spec.generate_value().astype(onp.float16))
        self.online_params = self.trg_params#self.network.init(next(self.rng), sample_input)

        # Build and initialize optimizer.
        self.optimizer = optix.adam(learning_rate)
        self.opt_state = self.optimizer.init(self.online_params)
        self.epsilon = epsilon
        self.train_steps = 0
        self.target_update_period = target_update_period or 500
        self.discount = discount or 0.99
        #  if use_double_q:
        #      self.update_q = DQNLearning.update_q_double
        #  else:
        #      self.update_q = DQNLearning.update_q
        self.update_q = DQNLearning.update_q

        if use_priority:
            self.experience = PriorityBuffer(observation_spec.shape[1],  action_spec.num_values, 1, 2**19)
        else:
            self.experience = ExperienceBuffer(observation_spec.shape[1], action_spec.num_values, 1, 2**19)
        self.importance_beta = importance_beta
        self.last_obs = onp.empty(observation_spec.shape)
        #  self.last_lm = np.empty(observation_spec.shape)

    def exploit(self, observation, legal_actions):
        actions = DQNPolicy.eval_policy(
            self.network, self.online_params, next(self.rng), observation, legal_actions)
        return jax.tree_util.tree_map(onp.array, actions)

    def explore(self, observation, legal_actions):
        _, actions = DQNPolicy.policy(
            self.network, self.online_params,
            self.epsilon(self.train_steps), next(self.rng),
            observation, legal_actions)
        return jax.tree_util.tree_map(onp.array, actions)

    #  def train(self, observations_tm1, actions_tm1,
    #                  observations_t, legal_moves_t, rewards_t,
    #                  terminal_t):
    #      """Train the agent.
    #
    #      Args:
    #          observations_tm1 -- observations at t-1.
    #          actions_tm1      -- actions at t-1.
    #          observations_t   -- observations at t.
    #          legal_moves_t    -- actions at t.
    #          rewards_t        -- rewards at t.
    #          terminal_t       -- terminal state at t?
    #      """
    #      self.online_params, self.opt_state = self.update_q(
    #          self.network,
    #          self.optimizer,
    #          self.online_params,
    #          self.trg_params,
    #          self.opt_state,
    #          observations_tm1, actions_tm1, observations_t, legal_moves_t,
    #          rewards_t, terminal_t, self.discount)
    #
    #      if self.train_steps % self.target_update_period == 0:
    #          self.trg_params = self.online_params
    #
    #      self.train_steps += 1
    def add_experience_first(self, observations, legal_moves, step_types):
        first_steps = step_types == 0
        self.last_obs[first_steps] = observations[first_steps]
        #  self.last_lm[first_steps] = legal_moves[first_steps]

    def add_experience(self, observations, legal_moves, actions, rewards, step_types):

        not_first_steps = step_types != 0
        self.experience.add_transitions(
            self.last_obs[not_first_steps],
            actions[not_first_steps].reshape((-1, 1)),
            rewards[not_first_steps].reshape((-1, 1)),
            observations[not_first_steps],
            legal_moves[not_first_steps],
            (step_types[not_first_steps] == 2).reshape((-1, 1)))
        self.last_obs[not_first_steps] = observations[not_first_steps]
        #  self.last_lm[not_first_steps] = legal_moves[not_first_steps]
        #  for _ in range(len(observations_tm1)):
        #      self.experience.add_new_sample(Transition(observations_tm1, actions_tm1, rewards_t, observations_t))

    #  def add_experience(self, observations_tm1, actions_tm1, rewards_t,
    #                     observations_t, legal_moves_t,
    #                     terminal_t):
    #
    #      self.experience.add_transitions(observations_tm1,
    #              actions_tm1, rewards_t, observations_t, legal_moves_t, terminal_t)
        #  for _ in range(len(observations_tm1)):
        #      self.experience.add_new_sample(Transition(observations_tm1, actions_tm1, rewards_t, observations_t))

    def update(self):
        batch_size = 256
        """Train the agent.

        Args:
            observations_tm1 -- observations at t-1.
            actions_tm1      -- actions at t-1.
            observations_t   -- observations at t.
            legal_moves_t    -- actions at t.
            rewards_t        -- rewards at t.
            terminal_t       -- terminal state at t?
        """

        #  if self.train_steps % 10 == 0:
        #  weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        if self.use_priority:
            sample_indices, weights_is, transitions = self.experience.sample_batch(batch_size, self.importance_beta)
        else:
            transitions = self.experience.sample(batch_size)
            weights_is = 
        
        #  print(sample_indices)
        #  else:
            #  transitions = self.experience.sample(batch_size)
        #  transitions = Transition()

        self.online_params, self.opt_state, tds = self.update_q(
            self.network,
            self.optimizer,
            self.online_params,
            self.trg_params,
            self.opt_state,
            transitions,
            #  observations_tm1, actions_tm1, observations_t, legal_moves_t,
            #  rewards_t, terminal_t,
            self.discount,
            weights_is,
            self.importance_beta)

        #  jax.vmap(jit_update_prio, in_axes=(None, 0, 0))(self.experience, sample_idxs, tds)
        #  if self.train_steps % 10 == 0:
        #  self.experience.update_priorities(sample_indices, onp.abs(tds))

        if self.train_steps % self.target_update_period == 0:
            self.trg_params = self.online_params

        self.train_steps += 1
