import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np

from .params import PPOParams, AgentType
from .vectorized_stacker import VectorizedObservationStacker
from .experience_buffer import ExperienceBuffer

import haiku as hk
import jax
import optax
import jax.numpy as jnp
import rlax
import chex

import collections
import pickle
from functools import partial
from typing import Tuple, List
from os.path import join as join_path

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")






################################## PPO Policy ##################################

# TODO Replace with Experience buffer
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):

	def __init__(
			self,
			state_dim,
			action_dim,
			action_std_init):

		super(ActorCritic, self).__init__()
		
		# Actor with Discrete action space
		self.actor = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, action_dim),
			nn.Softmax(dim=-1)
		)

		# Critic
		self.critic = nn.Sequential(
			nn.Linear(state_dim, 64),
			nn.Tanh(),
			nn.Linear(64, 64),
			nn.Tanh(),
			nn.Linear(64, 1)
		)

	def forward(self):
		raise NotImplementedError

	def act(self, state):
		"""
			An action is taken by passing the state through the actor network

			Args: 
				state	:	current state of the agent
			
			Returns: 
				action 	: 	Action selected by the actor
				action_logprob	: Log_probability of the selected action 

		
		"""

		# Pass the state through the actor network
		# to get the action probabilities
		action_probs = self.actor(state)
		
		# Make a categorical distribution of the 
		# action probabilities
		dist = Categorical(action_probs)

		# Sample an action fromt this distribution
		action = dist.sample()

		# Get the log_probability of this action
		action_logprob = dist.log_prob(action)

		return action.detach(), action_logprob.detach()

	def evaluate(self, state, action):
		"""
			Evaluate a state-action pair 

			Args: 
				state	: 	Next state of the environment 
				action	: 	Action taken in this state	
    
			Returns: 
				action_logprobs	:	Log probability of action		 
				state_values	:	values of state 
				dist_entropy	: 	Entropy of the action distribution
		"""
		
		# Get action probability by passing the state through the actor network 
		action_probs = self.actor(state)
		
		# Make a categorical distribution of actions
		dist = Categorical(action_probs)

		# get the log probabilities of actions
		action_logprobs = dist.log_prob(action)
		
		# Get the entropy of this distribution
		dist_entropy = dist.entropy()

		# Pass the state through the critic to get the state value
		state_values = self.critic(state)

		return action_logprobs, state_values, dist_entropy


class PPO:
	def __init__(
		self, 
		observation_spec,
		action_spec,
		params: PPOParams = PPOParams(),
		action_std_init=0.6,
		reward_shaper=None):

		self.type = AgentType().type
		self.params = params
		self.reward_shaper = reward_shaper
		# TODO Partition into a params class
		self.gamma = params.gamma
		self.eps_clip = params.eps_clip
		self.K_epochs = params.K_epochs

		self.observation_len = observation_spec.shape[1] * self.params.history_size
		# TODO Replace with Experience Buffer 
		self.buffer = ExperienceBuffer(
			# shape of observation
			observation_len=self.observation_len,
			capacity = params.experience_buffer_size  # buffer size
		)

		# We require vectorized observations from the game
		self.requires_vectorized_observation = lambda: True

		# Define the state and action dimensions
		self.state_dim = np.zeros(
					(observation_spec.shape[0], observation_spec.shape[1]
						* self.params.history_size),
					dtype=onp.float16
				)
		self.action_dim = action_spec.num_values

		# Define the Actor-Critic policy based on state and action dimensions
		self.policy = ActorCritic(
					state_dim=state_dim,
					action_dim=action_dim,
					action_std_init=action_std_init
				).to(device)

		# Define the optimizer using the learning rates for actor and critic networks
		self.optimizer = torch.optim.Adam([
			{'params': self.policy.actor.parameters(), 'lr': params.lr_actor},
			{'params': self.policy.critic.parameters(), 'lr': params.lr_critic}
		])

		# Initialize the old policy similar to the current policy
		self.policy_old = ActorCritic(
							state_dim = state_dim,
							action_dim = action_dim,
							action_std_init= action_std_init
						).to(device)
		
		# initialize the state_dict of the module that it is extending
		self.policy_old.load_state_dict(self.policy.state_dict())
		
		# Set the error type. NOTE Maybe can be extended to other losse ? 
		self.MseLoss = nn.MSELoss()

	# TODO change state to observations
	# TODO handle legal actions
	def exploit(self, observations):
		"""
			Select an action from the current state using the current policy

			Args:
				state	:	Current state of the agent 

			Returns:
				action	:	Selected action
		"""
		
		observations, legal_actions = observations[1]

		# Disable gradient since we will not be calling 
		# the backward function for selection
		with torch.no_grad():

			# Convert the state to the appropriate tensor
			observations = torch.FloatTensor(observations).to(device)

			# Act on the current policy to get the action 
			# and the log probabilities
			action, action_logprob = self.policy_old.act(observations)

		return (action.item(), action_logprob)

	# TODO can we use a different policy for exploration ?
	# TODO Change state to observations
	def explore(self, state):
		"""
		Select an action from the current state using the current policy

		Args:
			state	:	Current state of the agent 

		Returns:
			action	:	Selected action
		"""

		observations, legal_actions = observations[1]

		# Disable gradient since we will not be calling
		# the backward function for selection
		with torch.no_grad():

			# Convert the state to the appropriate tensor
			observations = torch.FloatTensor(observations).to(device)

			# Act on the current policy to get the action
			# and the log probabilities
			action, action_logprob = self.policy_old.act(observations)

		return (action.item(), action_logprob)

	# TODO Correct stuff
	def shape_rewards(
			self, 
			observations, 
			moves	) -> Union[ndarray, int]:
		"""
		Apply reward shaping function to list of HanabiObservation, HanabiMove
		if shaper object is defined at initialization

		Args
			observations : Vector of observations
			moves		 : Vector of moves that made those observations

		Returns
			A tuple of Shaped rewards and shaping type applied to those rewards
		"""
		if self.reward_shaper is not None:
			shaped_rewards, shape_type = self.reward_shaper.shape(observations[0],
																	moves)
			return np.array(shaped_rewards), np.array(shape_type)
		return (np.zeros(len(observations[0])), np.zeros(len(observations[0])))

	def update(
		self,
		factor=None,
		diversity=None	):
		"""
		Update the policy using the PPO formulation
		"""
		# Monte Carlo estimate of returns
		rewards = []
		discounted_reward = 0

		# Sample the rewards from the buffer and discount by one timestep if not terminal 
		# before adding
		for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
			if is_terminal:
				discounted_reward = 0
			discounted_reward = reward + (self.gamma * discounted_reward)
			rewards.insert(0, discounted_reward)

		# Normalizing the rewards
		rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
		rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

		# convert list to tensor
		old_states = torch.squeeze(torch.stack(
			self.buffer.states, dim=0)).detach().to(device)
		old_actions = torch.squeeze(torch.stack(
			self.buffer.actions, dim=0)).detach().to(device)
		old_logprobs = torch.squeeze(torch.stack(
			self.buffer.logprobs, dim=0)).detach().to(device)

		# Optimize policy for K epochs
		for _ in range(self.K_epochs):

			# Evaluating old actions and values
			logprobs, state_values, dist_entropy = self.policy.evaluate(
													state = old_states, 
													action = old_actions)

			# match state_values tensor dimensions with rewards tensor
			state_values = torch.squeeze(state_values)

			# Finding the ratio (pi_theta / pi_theta__old)
			ratios = torch.exp(logprobs - old_logprobs.detach())

			# Finding Surrogate Loss
			advantages = rewards - state_values.detach()
			surr1 = ratios * advantages
			surr2 = torch.clamp(
						ratios, 1-self.eps_clip,
						1+self.eps_clip	) * advantages

			# final loss of clipped objective PPO
			loss = -torch.min(surr1, surr2) + 0.5 * \
				self.MseLoss(state_values, rewards) - 0.01*dist_entropy

			# take gradient step using the backward function
			self.optimizer.zero_grad()
			loss.mean().backward()
			self.optimizer.step()

		# Copy new weights into old policy
		self.policy_old.load_state_dict(self.policy.state_dict())

		# TODO replace by experience buffer 
		# clear buffer
		self.clear_buffer()

	# TODO integrate log probabilities
	def add_experience(
			self, 
			observation_tm1: np.ndarray,
            action_tm1: Tuple[np.ndarray, np.ndarray],
			reward_t: np.ndarray,
			observation_t: np.ndarray,
			terminal_t: np.ndarray) -> None:

		"""
		Add transition batch to the experience buffer

		Args:
			observation_tm1	:	New Observation vector
			action_tm1		:	Tuple of actions nad log_probs
			reward_t		: 	rewards for the transitions 
			observation_t	:	Current Observation vector
			terminal_t		: 	Vector of terminal states 

		Returns 
			None
		
		"""

		obs_vec_tm1 = observations_tm1[1][0]
		obs_vec_t = observations_t[1][0]


		actions = action_tm1[0]
		log_probs = action_tm1[1]

		self.experience.add_transitions(
			observation_tm1=obs_vec_tm1,
			action_tm1=actions.reshape(-1, 1),
			log_probs_tm1=log_probs_tm1.reshape(-1, 1),
			reward_t=rewards_t,
			observation_t=obs_vec_t,
			terminal_t=terminal_t)

	def create_stacker(
				self, 
				obs_len, 
				n_states	):
		return VectorizedObservationStacker(self.params.history_size, obs_len, n_states)

	def save(self, checkpoint_path):
		torch.save(self.policy_old.state_dict(), checkpoint_path)

	def load(self, checkpoint_path):
		self.policy_old.load_state_dict(torch.load(
			checkpoint_path, map_location=lambda storage, loc: storage))
		self.policy.load_state_dict(torch.load(
			checkpoint_path, map_location=lambda storage, loc: storage))

	# TODO verify if this actually makes sense ?
	def clear_buffer(self):
		"""
			clear the buffer by re-allocating new space and clearing the old space
		"""
		del self.buffer
		
		self.buffer = ExperienceBuffer(
			# shape of observation
			observation_len=self.observation_len,
			capacity = self.params.experience_buffer_size  # buffer size
		)

	def add_experience_first(self, observations, step_types):
		pass
