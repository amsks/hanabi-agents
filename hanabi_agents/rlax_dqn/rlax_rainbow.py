"""
This file implements a DQNAgent.
"""
import collections
import pickle
from functools import partial
from typing import Tuple, List
from os.path import join as join_path
import timeit
import copy
import numpy as onp

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import rlax
import chex

# from .experience_buffer import ExperienceBuffer, sample_from_buffer
# from .priority_buffer import PriorityBuffer
from .experience_buffer_old import ExperienceBuffer
from .priority_buffer_old import PriorityBuffer
from .noisy_mlp import NoisyMLP
# from .noisy_mlp2 import NoisyMLP2
#from .mlp import MLP
from .params import RlaxRainbowParams
from .vectorized_stacker import VectorizedObservationStacker

from optax._src import combine
from optax._src import transform
from typing import NamedTuple, Any, Callable, Sequence, Optional, Union

DiscreteDistribution = collections.namedtuple(
    "DiscreteDistribution", ["sample", "probs", "logprob", "entropy"])

OptState = NamedTuple  # Transformation states are (possibly empty) namedtuples.
Params = Any  # Parameters are arbitrary nests of `jnp.ndarrays`.
Updates = Params  # Gradient updates are of the same type as parameters.

# Function used to initialise the transformation's state.
TransformInitFn = Callable[
    [Params],
    Union[OptState, Sequence[OptState]]]
# Function used to apply a transformation.
TransformUpdateFn = Callable[
    [Updates, OptState, Optional[Params]],
    Tuple[Updates, OptState]]

class GradientTransformation(NamedTuple):
  """Optax transformations consists of a function pair: (initialise, update)."""
  init: TransformInitFn
  update: TransformUpdateFn

def custom_adam(b1: float = 0.9, 
                b2: float = 0.999, 
                eps: float = 1e-8,
                eps_root: float = 0.0) -> GradientTransformation: 
    return combine.chain(
        transform.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root))

def apply_lr(lr, updates, state):
    updates = jax.tree_map(lambda g: -lr * g, updates)
    return updates

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
    def _apply_legal_boltzmann(probs, tau, legal):
        """Mix an arbitrary categorical distribution with a boltzmann distribution"""
        weighted_probs = probs / tau
        boltzmann_probs = jnp.where(
            legal, jnp.exp(weighted_probs - jnp.max(weighted_probs, axis=-1)[:, None]), 0.
        )
        boltzmann_probs = boltzmann_probs / jnp.sum(boltzmann_probs, axis=-1)[:, None]
        return boltzmann_probs

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
    def legal_softmax(tau=None):
        """An epsilon-greedy distribution with illegal probabilities set to zero"""

        def sample_fn(key: chex.Array,
                      preferences: chex.Array,
                      legal: chex.Array,
                      tau=tau):
            probs = DQNPolicy._apply_legal_boltzmann(preferences, tau, legal)
            return DQNPolicy._categorical_sample(key, probs)

        def probs_fn(preferences: chex.Array, legal: chex.Array, tau=tau):
            return DQNPolicy._apply_legal_boltzmann(preferences, tau, legal)

        def logprob_fn(sample: chex.Array,
                       preferences: chex.Array,
                       legal: chex.Array,
                       tau=tau):
            probs = DQNPolicy._apply_legal_boltzmann(preferences, tau, legal)
            return rlax.base.batched_index(jnp.log(probs), sample)

        def entropy_fn(preferences: chex.Array, legal: chex.Array, tau=tau):
            probs = DQNPolicy._apply_legal_boltzmann(preferences, tau, legal)
            return -jnp.nansum(probs * jnp.log(probs), axis=-1)

        return DiscreteDistribution(sample_fn, probs_fn, logprob_fn, entropy_fn)


    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 2))
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
            network    -- haiku Transformed network.
            net_params -- parameters (weights) of the network.
            key        -- key for categorical sampling.
            obs        -- observation.
            lm         -- one-hot encoded legal actions
        """
        
        print('compile policy')
        # compute q values
        # calculate q value from distributional output
        # by calculating mean of distribution
        if use_distribution:
            logits = network.apply(net_params, key, obs)
            probs = jax.nn.softmax(logits, axis=-1)
            q_vals = jnp.mean(probs * atoms, axis=-1)
            
        # q values equal network output
        else:
            q_vals = network.apply(net_params, key, obs)
        
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
            logits = network.apply(net_params, key, obs)
            probs = jax.nn.softmax(logits, axis=-1)
            q_vals = jnp.mean(probs * atoms, axis=-1)
            
        # q values equal network output
        else:
            q_vals = network.apply(net_params, key, obs)
        
        # mask q values of illegal actions
        q_vals = jnp.where(lms, q_vals, -jnp.inf)

        # select best action
        return rlax.greedy().sample(key, q_vals), jnp.max(q_vals, axis=1)

class DQNLearning:
    
    @staticmethod
    @partial(jax.jit, static_argnums=(0, 2, 11, 12))
    def update_q(   network, 
                    atoms, 
                    optimizer, 
                    lr, 
                    online_params, 
                    trg_params, 
                    opt_state,
                    transitions, 
                    discount_t, 
                    prios, 
                    beta_is, 
                    use_double_q, 
                    use_distribution, 
                    key_online, 
                    key_target, 
                    key_selector,
                    factor = 1,
                    diversity = 0): 
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
            q_logits_tm1 = network.apply(online_params, key_online, obs_tm1)
            q_logits_t = network.apply(trg_params, key_target, obs_t)
            q_logits_sel = network.apply(online_params, key_selector, obs_t)
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
            
            q_logits_tm1 = network.apply(online_params, key_online, obs_tm1)
            q_logits_t = network.apply(trg_params, key_target, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.categorical_q_learning,
                                   in_axes=(None, 0, 0, 0, 0, None, 0,))
            
            td_errors = batch_error(atoms[0], q_logits_tm1, a_tm1, r_t, discount_t, atoms[0], q_logits_t)
            return td_errors
        
        # calculate double q td loss (no distributional output)
        def double_q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            
            q_tm1 = network.apply(online_params, key_online, obs_tm1)
            q_t = network.apply(trg_params, key_target, obs_t)
            q_t_selector = network.apply(online_params, key_selector, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.double_q_learning,
                                   in_axes=(0, 0, 0, 0, 0, 0))
            
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t, q_t_selector)
            return td_errors**2
        
         # calculate q td loss (no distributional output)
        def q_td(online_params, trg_params, obs_tm1, a_tm1, r_t, obs_t, term_t, discount_t):
            
            q_tm1 = network.apply(online_params, key_online, obs_tm1)
            q_t = network.apply(trg_params, key_target, obs_t)
            
            # set discount to zero if state terminal
            term_t = term_t.reshape(r_t.shape)
            discount_t = jnp.where(term_t, 0, discount_t)
            
            batch_error = jax.vmap(rlax.q_learning,
                                   in_axes=(0, 0, 0, 0, 0,))
            
            td_errors = batch_error(q_tm1, a_tm1, r_t, discount_t, q_t)
            return td_errors**2

        def loss(   online_params, 
                    trg_params, 
                    obs_tm1, 
                    a_tm1, 
                    r_t, 
                    obs_t, 
                    term_t, 
                    discount_t, 
                    prios,
                    factor = 1,
                    diversity = 0   ):
            
            
            
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
            mean_loss = jnp.mean(batch_loss * weights_is + factor * diversity)
            if use_distribution:
                new_prios = jnp.abs(batch_loss)
            else: 
                new_prios = jnp.sqrt(batch_loss)
            return mean_loss, new_prios


        print('update')
        grad_fn = jax.grad(loss, has_aux=True)
        grads, new_prios = grad_fn(
            online_params, trg_params,
            transitions["observation_tm1"],
            transitions["action_tm1"][:, 0],
            transitions["reward_t"][:, 0],
            transitions["observation_t"],
            transitions["terminal_t"],
            discount_t,
            prios,
            factor,
            diversity
        )

        updates, opt_state_t = optimizer.update(grads, opt_state)
        updates = apply_lr(lr, updates, opt_state)
        online_params_t = optax.apply_updates(online_params, updates)
        return online_params_t, opt_state_t, new_prios


class DQNAgent:
    def __init__(
            self,
            observation_spec,
            action_spec,
            params: RlaxRainbowParams = RlaxRainbowParams(),
            reward_shaper = None,
            buffersizes = None,
            lrs = None,
            alphas = None,
            ):

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
        self.n_network = params.n_network
        # for evaluating absolute td errors near start of training
        self.drawn_td_abs = [[] for _ in range(self.n_network)]
        self.drawn_transitions = []
        self.random_transitions = []
        self.store_td = False
        if self.params.pbt == True:
            self.past_obs = onp.zeros((self.params.past_obs_size, observation_spec.shape[1]))
            self.past_lms = onp.zeros((self.params.past_obs_size, action_spec.num_values))

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
        # self.optimizer = optax.adam(params.learning_rate, eps=3.125e-5)
        if lrs is None:
            self.lr = onp.zeros(self.n_network)
            for i in range(self.n_network):
                self.lr[i] = params.learning_rate
        else:
            self.lr = onp.asarray(lrs)        
        self.optimizer = custom_adam(eps = 3.125e-5)
        parallel_opt_init = jax.vmap(self.optimizer.init, in_axes=(0,))
        self.opt_state = parallel_opt_init(self.online_params)
        
        #self.opt_state = self.optimizer.init(self.online_params)
        self.train_step = 0
        
        # Define the update function
        self.update_q = DQNLearning.update_q

        #define if same parameters or set of different parameters
        if buffersizes is None:
            self.buffersize = [params.experience_buffer_size for i in range(self.n_network)]
        else:
            self.buffersize = buffersizes
        if alphas is None:
            self.alpha = [params.priority_w for i in range(self.n_network)]
        else:
            self.alpha = alphas
        # Create Buffer (Priority Buffer or Experience Buffer)    
        if params.use_priority:
            self.buffer = [PriorityBuffer(
                self.buffersize[i],
                observation_spec.shape[1] * self.params.history_size,
                self.alpha[i]
            ) for i in range(self.n_network)]
        else:
            self.buffer = [ExperienceBuffer(
                self.buffersize[i],
                observation_spec.shape[1] * self.params.history_size
            ) for i in range(self.n_network)]
            
        self.requires_vectorized_observation = lambda: True

        # vmapped functions for parallel training of n_networks of agents
        self.parallel_update = jax.vmap(self.update_q, in_axes=(None, None, None, 0, 0, 0, 0, 
                                                                {"observation_tm1" : 0, "action_tm1" : 0, "reward_t" : 0, "observation_t" : 0, "terminal_t" : 0},
                                                                None, 0, None, None, None, 0, 0, 0))
        self.parallel_eval_exploit = jax.vmap(DQNPolicy.eval_policy, in_axes=(None, None, None, 0, 0, 0, 0))
        self.parallel_eval = jax.vmap(DQNPolicy.policy, in_axes=(None, None, None, None, 0, None, None, 0, 0, 0))


    def exploit(self, observations, eval = False):

        observations, legal_actions = observations[1]
        
        obs = observations.reshape(self.n_network, -1, observations.shape[1])
        vla = legal_actions.reshape(self.n_network, -1, legal_actions.shape[1])
        keys = onp.array([onp.random.randint(2147483647, size = 2, dtype='uint32') for _ in range(self.n_network)])
        
        
        actions, q_values = self.parallel_eval_exploit(self.network, self.params.use_distribution, self.atoms, 
                                self.online_params, keys, obs, vla)
        
        return (jax.tree_util.tree_map(onp.array, actions).flatten(),
                jax.tree_util.tree_map(onp.array, q_values).flatten())

    def explore(self, observations):
        if self.params.pbt == True and self.train_step % 300 == 0:
            self.past_obs = onp.roll(self.past_obs, -int(observations[1][0].shape[0]), axis=0)
            self.past_obs[-int(observations[1][0].shape[0]):, :] = observations[1][0]
            self.past_lms = onp.roll(self.past_lms, -int(observations[1][1].shape[0]), axis=0)
            self.past_lms[-int(observations[1][1].shape[0]):, :] = observations[1][1]

        observations, legal_actions = observations[1]

        obs = observations.reshape(self.n_network, -1, observations.shape[1])
        vla = legal_actions.reshape(self.n_network, -1, legal_actions.shape[1])
        keys = onp.array([onp.random.randint(2147483647, size = 2, dtype='uint32') for _ in range(self.n_network)])


        _, actions = self.parallel_eval(
            self.network, self.params.use_distribution, self.params.use_boltzmann_exploration,
            self.atoms, self.online_params, 
            self.params.epsilon(self.train_step), self.params.tau(self.train_step), 
            keys, obs, vla
        )

        return jax.tree_util.tree_map(onp.array, actions).flatten()
    
    def add_experience_first(self, observations, step_types):
        pass

    def add_experience(self, observations_tm1, actions_tm1, rewards_t, observations_t, term_t):

        obs_vec_tm1 = observations_tm1[1][0]
        obs_vec_t = observations_t[1][0]
        

        obs_len = int(obs_vec_tm1.shape[0]/self.n_network)

        for i in range(self.n_network):
            self.buffer[i].add(
                obs_vec_tm1[i*obs_len:(i+1)*obs_len],
                actions_tm1[i*obs_len:(i+1)*obs_len],
                rewards_t[i*obs_len:(i+1)*obs_len],
                obs_vec_t[i*obs_len:(i+1)*obs_len],
                term_t[i*obs_len:(i+1)*obs_len])

        
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
        
        if not self.params.fixed_weights:
            
            sample_indices, prios, c = [], [], []

            for i in range(self.n_network):
                _tra, _samp, _prio = self.buffer[i].sample(self.params.train_batch_size)
                sample_indices.append(_samp)
                prios.append(_prio)
                c.append(_tra._asdict())

            prios = onp.asarray(prios).reshape(self.n_network, self.params.train_batch_size)

            transitions = {}
            for key in c[0]:
                transitions[key] = onp.stack([b[key] for b in c], axis = 0)
            
            
            
            keys_online = onp.array([onp.random.randint(2147483647, size = 2, dtype='uint32') for _ in range(self.n_network)])
            keys_target = onp.array([onp.random.randint(2147483647, size = 2, dtype='uint32') for _ in range(self.n_network)])
            keys_sel = onp.array([onp.random.randint(2147483647, size = 2, dtype='uint32') for _ in range(self.n_network)])
            
            
            self.online_params, self.opt_state, tds = self.parallel_update(
                self.network,
                self.atoms,
                self.optimizer,
                self.lr,
                self.online_params,
                self.trg_params,
                self.opt_state,
                transitions,
                self.params.discount,
                prios,
                self.params.beta_is(self.train_step),
                self.params.use_double_q,
                self.params.use_distribution,
                keys_online,
                keys_target,
                keys_sel)
                        
            if self.params.use_priority:
                
                tds_abs = jax.tree_util.tree_map(onp.array, tds)
                for i in range(self.n_network):
                    self.buffer[i].update_priorities(sample_indices[i], tds_abs[i])

            if self.train_step % self.params.target_update_period == 0:
                self.trg_params = self.online_params
    
            self.train_step += 1
                            
    def create_stacker(self, obs_len, n_states):
        return VectorizedObservationStacker(self.params.history_size, 
                                            obs_len,
                                            n_states)

    def __repr__(self):
        return f"<rlax_dqn.DQNAgent(params={self.params})>"
    
    def save_weights(self, path, fname_part, only_weights=True, epochs_alive=None):
        """Save online and target network weights to the specified path
        added: save optimizer state"""
        
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_online.pkl"), 'wb') as of:
            pickle.dump(self.online_params, of)
        with open(join_path(path, "rlax_rainbow_" + fname_part + "_target.pkl"), 'wb') as of:
            pickle.dump(self.trg_params, of)
            
        if not only_weights:
            with open(join_path(path, "rlax_rainbow_" + fname_part + "_opt_state.pkl"), 'wb') as of:
                pickle.dump(jax.tree_util.tree_map(onp.array, self.opt_state), of)

            parameters = {'lr':self.lr, 'buffersize':self.buffersize, 'alpha':self.alpha, 'epoch_alive':epochs_alive}
            with open(join_path(path, "rlax_rainbow_" + fname_part + "parameters.pkl"), 'wb') as of:
                pickle.dump(parameters, of)
            # with open(join_path(path, "rlax_rainbow_" + fname_part + "_experience.pkl"), 'wb') as of:
            #     pickle.dump(self.buffer.serializable(), of)
    
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
                        experience_file=None,
                        parameter_file=None):
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
            self.train_step = self.opt_state[0].count[0]
        # experience buffer
        if experience_file is not None:
            with open(experience_file, 'rb') as iwf:
                self.buffer.load(pickle.load(iwf))
        epoch_alive = None
        if parameter_file is not None:
            with open(paremter_file, 'rb') as iwf:
                parameters = pickle.load(iwf)
                self.buffersize = paramters['buffersize']
                self.alpha = paramters['alpha']
                self.lr = parameters['lr']
                for i, size in enumerate(self.buffersize):
                    self.buffer[i].change_buffersize(size)
                    self.buffer.alpha = self.alpha[i]
                epochs_alive = parameters['epoch_alive']
        return epoch_alive
                
    def get_buffer_tds(self):
        if self.params.use_priority:
            # index_list = range(self.buffer.size)
            # return self.buffer.get_tds(index_list)
            return [0, 0]
        else:
            return None
         
    def get_drawn_tds(self, reset=True, deactivate=True):
        tds = onp.array(self.drawn_td_abs)
        transitions = copy.deepcopy(self.drawn_transitions)
        random_transitions = copy.deepcopy(self.random_transitions)
        if reset:
            self.drawn_transitions.clear()
            self.random_transitions.clear()
            for i in range(self.n_network):
                self.drawn_td_abs[i].clear()
        self.store_td = not deactivate
        # return tds, transitions, random_transitions
        return [0,0,0]
     
    def get_stochasticity(self):
         
        def process_weights(w):
            return onp.mean(onp.abs(w['w_sigma']), axis=(1,2))
         
        if self.params.use_noisy_network:
            weights = hk.data_structures.to_mutable_dict(self.online_params)
            return [process_weights(w) for w in weights.values()]
        else:
            return None
