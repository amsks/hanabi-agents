from typing import Tuple
import numpy as np
import pickle
from .transition import Transition

class ExperienceBuffer:
    """ExperienceBuffer stores transitions for training"""

    def __init__(self, observation_len: int, capacity: int):
        self._obs_tm1_buf = np.empty((capacity, observation_len), dtype=np.byte)
        self._act_tm1_buf = np.empty((capacity, 1), dtype=np.byte)
        self._obs_t_buf = np.empty((capacity, observation_len), dtype=np.byte)
        self._rew_t_buf = np.empty((capacity, 1), dtype=np.float64)
        self._terminal_t_buf = np.empty((capacity, 1), dtype=bool)
        self._sample_range = np.arange(0, capacity, dtype=np.int)
        self.capacity = capacity
        self.oldest_entry = 0
        self.size = 0

    def get_update_indices(self, batch_size):
        """
        Get indices of oldest entries in buffer.
        """
        max_entry = self.oldest_entry + batch_size
        if max_entry <= self.capacity:
            return list(range(self.oldest_entry, max_entry))
        part1 = list(range(self.oldest_entry, self.capacity)) # end of buffer
        part2 = list(range(max_entry - self.capacity)) # start of buffer
        return  + part1 + part2

    def add_transitions(self,
                        observation_tm1: np.ndarray,
                        action_tm1: np.ndarray,
                        reward_t: np.ndarray,
                        observation_t: np.ndarray,
                        terminal_t: np.ndarray):
        """Add a transition to buffer.
        Args:
            observation_tm1 -- source observation. shape (batch_size, observation_len)
            action_tm1      -- action taken from source to destination state. shape (batch_size, 1)
            observation_t   -- destination observation. batch of shape (batch_size, observation_len)
            reward_t        -- reward for getting from source to destination state. shape (batch_size, 1)
            terminal_t      -- flag showing whether the destination state is terminal. shape (batch_size, 1)
        """
        batch_size = len(observation_tm1)
        
        # new batch is written into middle of buffer
        if self.oldest_entry + batch_size <= self.capacity:
            self._obs_tm1_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = observation_tm1
            self._act_tm1_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = action_tm1
            self._obs_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = observation_t
            self._rew_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = reward_t
            self._terminal_t_buf[self.oldest_entry : self.oldest_entry + batch_size, :] = terminal_t
            self.size = max(self.size, self.oldest_entry + batch_size)
            self.oldest_entry = (self.oldest_entry + batch_size) % self.capacity
            
        # while writing batch into the buffer, end of buffer is reached
        else:
            tail = self.oldest_entry + batch_size - self.capacity
            self._obs_tm1_buf[self.oldest_entry:, :] = observation_tm1[:batch_size - tail]
            self._act_tm1_buf[self.oldest_entry:, :] = action_tm1[:batch_size - tail]
            self._obs_t_buf[self.oldest_entry:, :] = observation_t[:batch_size - tail]
            self._rew_t_buf[self.oldest_entry:, :] = reward_t[:batch_size - tail]
            self._terminal_t_buf[self.oldest_entry:, :] = terminal_t[:batch_size - tail]
            self._obs_tm1_buf[:tail, :] = observation_tm1[-tail:]
            self._act_tm1_buf[:tail, :] = action_tm1[-tail:]
            self._obs_t_buf[:tail, :] = observation_t[-tail:]
            self._rew_t_buf[:tail, :] = reward_t[-tail:]
            self._terminal_t_buf[:tail, :] = terminal_t[-tail:]
            self.oldest_entry = tail
            self.size = self.capacity

    def __getitem__(self, indices):
        return Transition(
            self._obs_tm1_buf[indices], 
            self._act_tm1_buf[indices],
            self._rew_t_buf[indices], 
            self._obs_t_buf[indices],
            self._terminal_t_buf[indices])
        
    def sample_batch(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample <batch_size> transitions from the ExperienceBuffer.
        """
        indices = np.random.choice(self._sample_range[:self.size], size=batch_size)
        return self[indices]
    
    def serializable(self):    
        """
        Get pickable representation of Replay Buffer.
        """  
        lst_serialize = [self._obs_tm1_buf, 
                         self._act_tm1_buf, 
                         self._obs_t_buf,
                         self._rew_t_buf, 
                         self._terminal_t_buf,
                         self._sample_range, 
                         self.oldest_entry, 
                         self.capacity, 
                         self.size]
        return lst_serialize
    
    def load(self, lst_serializable):
        """
        Load pickable representation of Replay Buffer. Inverse function of serializable
        """
        self._obs_tm1_buf = lst_serializable[0]
        self._act_tm1_buf = lst_serializable[1]
        self._obs_t_buf = lst_serializable[2]
        self._rew_t_buf = lst_serializable[3]
        self._terminal_t_buf = lst_serializable[4]
        self._sample_range = lst_serializable[5]
        self.oldest_entry = lst_serializable[6]
        self.capacity = lst_serializable[7]
        self.size = lst_serializable[8]