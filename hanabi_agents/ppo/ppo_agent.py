"""
    This is the code for the ppo agent to be used in the Hanabi Game. 

"""

# TODO Trim down the imports

from typing import Tuple, Union, List
from numpy import ndarray
from hanabi_learning_environment.pyhanabi_pybind import HanabiMove, HanabiObservation
from hanabi_multiagent_framework.observation_stacker import ObservationStacker


import collections
import pickle
from functools import partial
from typing import Tuple, List
from os.path import join as join_path

import numpy as np

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import rlax
import chex


import gym
import sys
import torch

from .experience_buffer import ExperienceBuffer

# from arguments import get_args
from .ppo import PPO, EvalPolicy
from .network import FeedForwardNN
from .vectorized_stacker import VectorizedObservationStacker
# from eval_policy import eval_policy



class PPOAgent:
  def init(
    self,
    observation_spec,
    action_spec,
    params=None,
    actor_model: str = '',
    critic_model: str = '', 
    max_train_steps: int = 200_000_000  ):
    
    self.actor_model = actor_model
    self.critic_model = critic_model
    self.params = params


    # define the shape of the input and outputs and initialize the network 
    input_placeholder = np.zeros(
        (observation_spec.shape[0], observation_spec.shape[1]
         * self.params.history_size),
        dtype=np.float16
    )
    
    input_shape = input_placeholder
    output_shape = action_spec.num_values
    
    self.network = FeedForwardNN(
      in_dim=input_shape,
      out_dim=output_shape
    )
    
    # Use the network too paramterize the PPO policy
    self.model = PPO(
      policy_class=FeedForwardNN, 
      observation_spec=observation_spec,
      action_spec=action_spec,
      **self.params  )
    
    self.eval_policy = EvalPolicy
    
    self.experience = ExperienceBuffer(
        observation_len = observation_spec.shape[1] * 
                                  self.params.history_size,  # shape of observation
        capacity = params.train_batch_size  # buffer size
    )

    # The network will use vectorized observation
    self.requires_vectorized_observation = lambda: True
    
    # Set the Start of training to 0 and initialize the maximum number of steps
    self.train_step = 0
    self.max_train_steps = max_train_steps
  

  def explore(
    self, 
    observations: Union[List[HanabiObservation],
                        Tuple[List[HanabiObservation], Tuple[ndarray, ndarray]]]  
    ) -> Union[ndarray, List[HanabiMove]]:  
    """
        Get action from the network
    """    
    observations, legal_actions = observations[1]
    
    return model.get_action(observations)
  
  def exploit(
    self, 
    observations: Union[Tuple[ndarray], ndarray]) -> ndarray:
    """
      Get action from the network
    """
    observations, legal_actions = observations[1]
    
    return model.get_action(observations)
  
  
  def add_experience_first(
    self, 
    observations: ndarray,
    step_types: ndarray ) -> None:
    pass 
  
  def add_experience(
    self,
    observations_tm1,       # Observation at t = t+1
    actions_tm1,            # Actions at t = t+1
    rewards_t,              # Rewards at t = t+1
    observations_t,         # Observations at t = t
    term_t                    # Terminated episodes at t = t + 1  
    
    ) -> None:      
    """
        Add transition batch to experience buffer
    """

    obs_vec_tm1 = observations_tm1[1][0]
    obs_vec_t = observations_t[1][0]

    self.experience.add_transitions(
        obs_vec_tm1,
        actions_tm1.reshape(-1, 1),
        rewards_t,
        obs_vec_t,
        term_t
    )


  def shape_rewards(
    self,
    observations,
    moves) -> Union[ndarray, int]:
    
    pass 
  
  def create_stacker(
    self,
    obs_len,
    n_states) -> ObservationStacker:
      return VectorizedObservationStacker(self.params.history_size, obs_len, n_states) 
  
  
  def update( self ):
    """
      Make one training step
    """
    
    # This should call a function in the policy that executes a rollout 
    # self.model.learn(
    #     batch_obs,
    #     batch_acts,
    #     batch_log_probs,
    #     batch_rtgs,
    #     batch_lens
    # )
    pass
  


