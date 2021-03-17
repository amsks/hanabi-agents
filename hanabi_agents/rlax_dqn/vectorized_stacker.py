import numpy as np


class VectorizedObservationStacker(object):
    """Class for stacking agent observations."""
  
  
    def __init__(self, history_size, observation_size, n_states):
        
        """Initializer for observation stacker.
        #TODO: number of states instead of self.num_players and list to np.array
        Args:
        history_size: int, number of time steps to stack.
        observation_size: int, size of observation vector on one time step.
        num_players: int, number of players.
        """
        
        self._history_size = history_size
        self._observation_size = observation_size
        self._n_states = n_states
        self._obs_stacks = np.zeros((self._n_states, self._history_size + 1, self._observation_size))
        self._size = np.zeros((self._n_states), dtype=int)

    def add_observation(self, observation):
        """add vectorized observation to stack
    
        Args:
          observation: observation vector for current player.
        """
        #print('obs has shape', observation.shape)
        self._obs_stacks = np.roll(self._obs_stacks, -1, axis = 1)
        self._obs_stacks[:, -1, :] = observation
        self._size += 1
        
    def get_current_obs(self):
        return self._obs_stacks[:, -self._history_size:, :].reshape(self._n_states, self._history_size * self._observation_size)

    def get_observation_stack_t(self):
        """Returns the stacked observation for current player."""
        return self._obs_stacks[:, -self._history_size:, :].reshape(-1, self._history_size * self._observation_size)

    def get_observation_stack_tm1(self):
        """Returns the stacked observation for current player shifted by the temporal difference."""
        return self._obs_stacks[:, :self._history_size, :].reshape(-1, self._history_size * self._observation_size)

    def reset(self, indices=None):
        """Resets the observation stacks to all zero."""
        if indices is None:
            self._size.fill(0)
            self._obs_stacks.fill(0.0)
        else:
            self._size[indices] = 0
            self._obs_stacks[indices] = 0.0
            
    def reset_history(self, indices=None):
        """ 
        clear history of states indicated by indices from stack
        only the current observation remains in stack
        """
        if indices is None:
            self._size[indices] = 1
            self._obs_stacks[indices, :-1, : ] = 0.0
        else:
            self._size.fill(1)
            self._obs_stacks[:, :-1, : ] = 0.0

    @property
    def history_size(self):
        """Returns number of steps to stack."""
        return self._history_size

    @property
    def observation_size(self):
        """Returns the size of the observation vector after history stacking."""
        return self._observation_size * self._history_size
    
    @property 
    def size(self):
        return self.size




    
