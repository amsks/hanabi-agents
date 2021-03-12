import numpy as np
import typing
from .transition import Transition


def sample_from_buffer(buffer_state, indices):

    return Transition(
        buffer_state["obs_tm1"][indices], 
        buffer_state["act_tm1"][indices],
        buffer_state["rew_t"][indices],
        buffer_state["obs_t"][indices], 
        buffer_state["term_t"][indices]
    )   
    

class ExperienceBuffer:
    
    def __init__(self, capacity: int, obs_len: int, n_network: int):
        self.capacity = capacity
        self.size = 0
        self.next_entry = 0
        self.n_network = n_network
        
        self.obs_tm1_buf = np.empty((n_network, self.capacity, obs_len), dtype=np.byte)
        self.act_tm1_buf = np.empty((n_network, self.capacity, 1), dtype=np.byte)
        self.obs_t_buf = np.empty((n_network, self.capacity, obs_len), dtype=np.byte)
        self.rew_t_buf = np.empty((n_network, self.capacity, 1), dtype=np.float64)
        self.terminal_t_buf = np.empty((n_network, self.capacity, 1), dtype=bool)
        
        self.sample_range = np.arange(0, self.capacity, dtype=np.int)
        
    def buffer_state(self):
        return {"obs_tm1": self.obs_tm1_buf, "act_tm1": self.act_tm1_buf,
                "obs_t": self.obs_t_buf, "rew_t": self.rew_t_buf, "term_t": self.terminal_t_buf}
        
    def get_update_indices(self, batch_size):
        if self.next_entry + batch_size <= self.capacity:
            return list(range(self.next_entry, self.next_entry + batch_size))
        tail = batch_size - self.capacity + self.next_entry
        return list(range(self.next_entry, self.capacity)) + list(range(0, tail))
        
    def add(self, 
            observation_tm1: np.ndarray,
            action_tm1: np.ndarray,
            reward_t: np.ndarray,
            observation_t: np.ndarray,
            terminal_t: np.ndarray):
        
        batch_size = observation_tm1.shape[1]
        
        # buffer is not reached
        if self.next_entry + batch_size <= self.capacity:
            
            self.obs_tm1_buf[:, self.next_entry : self.next_entry + batch_size, :] = observation_tm1
            self.act_tm1_buf[:, self.next_entry : self.next_entry + batch_size, :] = action_tm1
            self.obs_t_buf[:, self.next_entry : self.next_entry + batch_size, :] = observation_t
            self.rew_t_buf[:, self.next_entry : self.next_entry + batch_size, :] = reward_t
            self.terminal_t_buf[:, self.next_entry : self.next_entry + batch_size, :] = terminal_t
            if self.next_entry + batch_size == self.capacity:
                self.size = self.capacity
            else:
                self.size = max(self.next_entry + batch_size, self.size)
            self.next_entry = (self.next_entry + batch_size) % self.capacity
        
        # handle the case when at the end of the buffer    
        else:
             
            tail = self.next_entry + batch_size - self.capacity
            self.obs_tm1_buf[:, self.next_entry:, :] = observation_tm1[:batch_size - tail]
            self.act_tm1_buf[:, self.next_entry:, :] = action_tm1[:batch_size - tail]
            self.obs_t_buf[:, self.next_entry:, :] = observation_t[:batch_size - tail]
            self.rew_t_buf[:, self.next_entry:, :] = reward_t[:batch_size - tail]
            self.terminal_t_buf[:, self.oldest_entry:, :] = terminal_t[:batch_size - tail]
            self.obs_tm1_buf[:, :tail, :] = observation_tm1[-tail:]
            self.act_tm1_buf[:, :tail, :] = action_tm1[-tail:]
            self.obs_t_buf[:, :tail, :] = observation_t[-tail:]
            self.rew_t_buf[:, :tail, :] = reward_t[-tail:]
            self.terminal_t_buf[:, :tail, :] = terminal_t[-tail:]
            self.next_entry = tail
            self.size = self.capacity
            
            
    def __getitem__(self, indices):
        
        def get_from_buffer(buf):
            return np.array([b[idx] for b, idx in zip(buf, indices)])
        
        samples = map(
            get_from_buffer, 
            [self.obs_tm1_buf, self.act_tm1_buf, self.rew_t_buf, self.obs_t_buf, self.terminal_t_buf]
        )
        
        return Transition(*samples)
    
    def sample(self, batch_size):
        indices = np.random.choice(self.sample_range[:self.size], size=(self.n_network, batch_size))
        return (
            self[indices],
            indices,
            onp.ones((self.n_network, self.params.train_batch_size)) # prios
        ) 
    
    def serializable(self):      
        lst_serialize = [self.obs_tm1_buf, 
                         self.act_tm1_buf, 
                         self.obs_t_buf,
                         self.rew_t_buf, 
                         self.terminal_t_buf,
                         self.sample_range, 
                         self.next_entry, 
                         self.capacity, 
                         self.size,
                         self.n_network]
        return lst_serialize
      
    def load(self, lst_serializable):
        self.obs_tm1_buf = lst_serializable[0]
        self.act_tm1_buf = lst_serializable[1]
        self.obs_t_buf = lst_serializable[2]
        self.rew_t_buf = lst_serializable[3]
        self.terminal_t_buf = lst_serializable[4]
        self.sample_range = lst_serializable[5]
        self.oldest_entry = lst_serializable[6]
        self.capacity = lst_serializable[7]
        self.size = lst_serializable[8]
        self.n_network = lst_serializable[9]
    
# from typing import Tuple
# import numpy as np
# import pickle
# from .transition import Transition
# import jax
#  
# class ExperienceBuffer:
#     """ExperienceBuffer stores transitions for training"""
#  
#     def __init__(self, observation_len: int, action_len: int, reward_len: int, capacity: int, n_network: int):
#         self._obs_tm1_buf = np.empty((n_network, capacity, observation_len), dtype=np.byte)
#         self._act_tm1_buf = np.empty((n_network, capacity, 1), dtype=np.byte)
#         self._obs_t_buf = np.empty((n_network, capacity, observation_len), dtype=np.byte)
#         self._lms_t_buf = np.empty((n_network, capacity, action_len), dtype=np.byte)
#         self._rew_t_buf = np.empty((n_network, capacity, reward_len), dtype=np.float64)
#         self._terminal_t_buf = np.empty((n_network, capacity, 1), dtype=bool)
#         self._sample_range = np.arange(0, capacity, dtype=np.int)
#         #  self.full = False
#         self.oldest_entry = 0
#         self.capacity = capacity
#         self.size = 0
#         self.n_network = n_network
#  
#     def get_update_indices(self, batch_size):
#         if self.oldest_entry + batch_size <= self.capacity:
#             return list(range(self.oldest_entry, self.oldest_entry + batch_size))
#         return list(range(self.oldest_entry, self.capacity)) + list(range(0, batch_size - self.capacity + self.oldest_entry))
#  
#     def add_transitions(self,
#                         observation_tm1: np.ndarray,
#                         action_tm1: np.ndarray,
#                         reward_t: np.ndarray,
#                         observation_t: np.ndarray,
#                         legal_moves_t: np.ndarray,
#                         terminal_t: np.ndarray):
#         """Add a transition to buffer.
#  
#         Args:
#             observation_tm1 -- source observation. batch of shape (batch_size, observation_len)
#             action_tm1      -- action taken from source to destination state.
#                                batch of shape (batch_size, 1)
#             observation_t   -- destination observation. batch of shape (batch_size, observation_len)
#             legal_moves_t   -- actions that can be taken from destination observation.
#                                batch of shape (batch_size, max_moves)
#             reward_t        -- reward for getting from source to destination state.
#                                batch of shape (batch_size, 1)
#             terminal_t      -- flag showing whether the destination state is terminal.
#                                batch of shape (batch_size, 1)
#         """
#         batch_size = observation_tm1.shape[1]#len(observation_tm1)
#         if self.oldest_entry + batch_size <= self.capacity:
#             self._obs_tm1_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = observation_tm1
#             self._act_tm1_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = action_tm1
#             self._obs_t_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = observation_t
#             self._lms_t_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = legal_moves_t
#             self._rew_t_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = reward_t
#             self._terminal_t_buf[:, self.oldest_entry : self.oldest_entry + batch_size, :] = terminal_t
#             if self.oldest_entry + batch_size == self.capacity:
#                 self.size = self.capacity
#             self.oldest_entry = (self.oldest_entry + batch_size) % self.capacity
#             self.size = max(self.size, self.oldest_entry)
#         else:
#             # handle the case when at the end of the buffer
#             tail = self.oldest_entry + batch_size - self.capacity
#             self._obs_tm1_buf[:, self.oldest_entry:, :] = observation_tm1[:batch_size - tail]
#             self._act_tm1_buf[:, self.oldest_entry:, :] = action_tm1[:batch_size - tail]
#             self._obs_t_buf[:, self.oldest_entry:, :] = observation_t[:batch_size - tail]
#             self._lms_t_buf[:, self.oldest_entry:, :] = legal_moves_t[:batch_size - tail]
#             self._rew_t_buf[:, self.oldest_entry:, :] = reward_t[:batch_size - tail]
#             self._terminal_t_buf[:, self.oldest_entry:, :] = terminal_t[:batch_size - tail]
#             self._obs_tm1_buf[:, :tail, :] = observation_tm1[-tail:]
#             self._act_tm1_buf[:, :tail, :] = action_tm1[-tail:]
#             self._obs_t_buf[:, :tail, :] = observation_t[-tail:]
#             self._lms_t_buf[:, :tail, :] = legal_moves_t[-tail:]
#             self._rew_t_buf[:, :tail, :] = reward_t[-tail:]
#             self._terminal_t_buf[:, :tail, :] = terminal_t[-tail:]
#             self.oldest_entry = tail
#             self.size = self.capacity
#  
#     def __getitem__(self, indices):
#         return Transition(
#             self._obs_tm1_buf[indices], self._act_tm1_buf[indices],
#             self._rew_t_buf[indices], self._obs_t_buf[indices],
#             self._lms_t_buf[indices], self._terminal_t_buf[indices])
#      
#     def serializable(self):      
#         lst_serialize = [self._obs_tm1_buf, 
#                          self._act_tm1_buf, 
#                          self._obs_t_buf,
#                          self._lms_t_buf, 
#                          self._rew_t_buf, 
#                          self._terminal_t_buf,
#                          self._sample_range, 
#                          self.oldest_entry, 
#                          self.capacity, 
#                          self.size]
#         return lst_serialize
#      
#     def load(self, lst_serializable):
#         self._obs_tm1_buf = lst_serializable[0]
#         self._act_tm1_buf = lst_serializable[1]
#         self._obs_t_buf = lst_serializable[2]
#         self._lms_t_buf = lst_serializable[3]
#         self._rew_t_buf = lst_serializable[4]
#         self._terminal_t_buf = lst_serializable[5]
#         self._sample_range = lst_serializable[6]
#         self.oldest_entry = lst_serializable[7]
#         self.capacity = lst_serializable[8]
#         self.size = lst_serializable[9]
#  
#     def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
#                                                np.ndarray, np.ndarray]:
#         """Sample <batch_size> transitions from the ExperienceBuffer.
#  
#         Returns (observation{batch_size, observation_len}, action{batch_size, 1},
#                  reward{batch_size, 1}, q_vals{batch_size, max_moves})
#         """
#         def sample_single(buffer, indices):
#             return buffer[indices]
#          
#         sample_parallel = jax.vmap(sample_single, in_axes=(0,0))
#         indices = np.random.choice(self._sample_range[:self.size], size=(self.n_network, self.batch_size))
#                                     
#         return sample_parallel(self, indices)#self[indices]
