from typing import Tuple
import numpy as np
import pickle
from .transition import Transition

class ExperienceBuffer:
    """ExperienceBuffer stores transitions for training"""

    def __init__(self, observation_len: int, action_len: int, reward_len: int, capacity: int):
        self._obs_tm1_buf = np.empty((capacity, observation_len), dtype=np.byte)
        self._act_tm1_buf = np.empty((capacity, 1), dtype=np.byte)
        self._obs_t_buf = np.empty((capacity, observation_len), dtype=np.byte)
        self._lms_t_buf = np.empty((capacity, action_len), dtype=np.byte)
        self._rew_t_buf = np.empty((capacity, reward_len), dtype=np.float64)
        self._terminal_t_buf = np.empty((capacity, 1), dtype=bool)
        self._sample_range = np.arange(0, capacity, dtype=np.int)
        #  self.full = False
        self.oldest_entry = 0
        self.capacity = capacity
        self.size = 0

    def get_update_indices(self, batch_size):
        if self.oldest_entry + batch_size <= self.capacity:
            return list(range(self.oldest_entry, self.oldest_entry + batch_size))
        return list(range(self.oldest_entry, self.capacity)) + list(range(0, batch_size - self.capacity + self.oldest_entry))

    def add_transitions(self,
                        observation_tm1: np.ndarray,
                        action_tm1: np.ndarray,
                        reward_t: np.ndarray,
                        observation_t: np.ndarray,
                        legal_moves_t: np.ndarray,
                        terminal_t: np.ndarray):
        """Add a transition to buffer.

        Args:
            observation_tm1 -- source observation. batch of shape (batch_size, observation_len)
            action_tm1      -- action taken from source to destination state.
                               batch of shape (batch_size, 1)
            observation_t   -- destination observation. batch of shape (batch_size, observation_len)
            legal_moves_t   -- actions that can be taken from destination observation.
                               batch of shape (batch_size, max_moves)
            reward_t        -- reward for getting from source to destination state.
                               batch of shape (batch_size, 1)
            terminal_t      -- flag showing whether the destination state is terminal.
                               batch of shape (batch_size, 1)
        """
        batch_size = len(observation_tm1)
        if self.oldest_entry + batch_size <= self.capacity:
            self._obs_tm1_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = observation_tm1
            self._act_tm1_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = \
                action_tm1
            self._obs_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = observation_t
            self._lms_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = legal_moves_t
            self._rew_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = \
                reward_t
            self._terminal_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = \
                terminal_t
            if self.oldest_entry + batch_size == self.capacity:
                self.size = self.capacity
            self.oldest_entry = (self.oldest_entry + batch_size) % self.capacity
            self.size = max(self.size, self.oldest_entry)
        else:
            # handle the case when at the end of the buffer
            tail = self.oldest_entry + batch_size - self.capacity
            self._obs_tm1_buf[self.oldest_entry:, :] = observation_tm1[:batch_size - tail]
            self._act_tm1_buf[self.oldest_entry:, :] = \
                action_tm1[:batch_size - tail]
            self._obs_t_buf[self.oldest_entry:, :] = observation_t[:batch_size - tail]
            self._lms_t_buf[self.oldest_entry:, :] = legal_moves_t[:batch_size - tail]
            self._rew_t_buf[self.oldest_entry:, :] = \
                reward_t[:batch_size - tail]
            self._terminal_t_buf[self.oldest_entry:, :] = \
                terminal_t[:batch_size - tail]
            self._obs_tm1_buf[:tail, :] = observation_tm1[-tail:]
            self._act_tm1_buf[:tail, :] = action_tm1[-tail:]
            self._obs_t_buf[:tail, :] = observation_t[-tail:]
            self._lms_t_buf[:tail, :] = legal_moves_t[-tail:]
            self._rew_t_buf[:tail, :] = reward_t[-tail:]
            self._terminal_t_buf[:tail, :] = terminal_t[-tail:]
            self.oldest_entry = tail
            self.size = self.capacity

    def __getitem__(self, indices):
        return Transition(
            self._obs_tm1_buf[indices], self._act_tm1_buf[indices],
            self._rew_t_buf[indices], self._obs_t_buf[indices],
            self._lms_t_buf[indices], self._terminal_t_buf[indices])
    
    def serializable(self):      
        lst_serialize = [self._obs_tm1_buf, 
                         self._act_tm1_buf, 
                         self._obs_t_buf,
                         self._lms_t_buf, 
                         self._rew_t_buf, 
                         self._terminal_t_buf,
                         self._sample_range, 
                         self.oldest_entry, 
                         self.capacity, 
                         self.size]
        return lst_serialize
    
    def load(self, lst_serializable):
        self._obs_tm1_buf = lst_serializable[0]
        self._act_tm1_buf = lst_serializable[1]
        self._obs_t_buf = lst_serializable[2]
        self._lms_t_buf = lst_serializable[3]
        self._rew_t_buf = lst_serializable[4]
        self._terminal_t_buf = lst_serializable[5]
        self._sample_range = lst_serializable[6]
        self.oldest_entry = lst_serializable[7]
        self.capacity = lst_serializable[8]
        self.size = lst_serializable[9]

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                               np.ndarray, np.ndarray]:
        """Sample <batch_size> transitions from the ExperienceBuffer.

        Returns (observation{batch_size, observation_len}, action{batch_size, 1},
                 reward{batch_size, 1}, q_vals{batch_size, max_moves})
        """
        indices = np.random.choice(self._sample_range[:self.size], size=batch_size)
        return self[indices]
