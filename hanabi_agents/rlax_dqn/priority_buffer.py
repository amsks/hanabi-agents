import numpy as onp
from sum_tree import SumTreef as SumTree
from .experience_buffer import ExperienceBuffer

class PriorityBuffer(ExperienceBuffer):
    """
    Extension of ExperienceBuffer, enables proportional prioritization of transitions.
    """
    def __init__(self, observation_len: int, capacity: int, alpha: int = 0.6):
        super(PriorityBuffer, self).__init__(observation_len, capacity)
        self.sum_tree = SumTree(capacity)
        self.max_priority = alpha if alpha > 0 else 1
        self.min_priority = alpha if alpha > 0 else 1
        self.alpha = alpha

    def add_transitions(self,
                        observation_tm1: onp.ndarray,
                        action_tm1: onp.ndarray,
                        reward_t: onp.ndarray,
                        observation_t: onp.ndarray,
                        terminal_t: onp.ndarray):
        """
        Add a batch of transitions to buffer and initialize replay priority.
        """
        
        batch_size = len(observation_tm1)
        indices = self.get_update_indices(batch_size)
        # new observation have highest priority
        priorities = [self.max_priority for _ in range(batch_size)]
        # update the priorities in the sum tree
        self.sum_tree.update_values(indices, priorities)
        # add observations to buffer
        super(PriorityBuffer, self).add_transitions(
            observation_tm1, action_tm1, reward_t, observation_t, terminal_t
        )

    def sample_batch(self, batch_size):
        """
        Sample a batch of transitions from replay buffer.
        Transitions are selected according to proportional prioritization.
        """
        
        # sampling for proportional prioritization
        # divide the range[0, 1] into batches and sample key from each batch
        keys = onp.linspace(1. / batch_size, 1, batch_size)
        keys -= onp.random.uniform(size=(batch_size,), high=1./batch_size)
        # use the key to retrieve indices (key=1 corresponds to tree root value)
        indices = self.sum_tree.get_indices(keys)
        # get priorities from sum tree and apply softmax normalization
        prios = (onp.array(self.sum_tree.get_values(indices))+ 1e-10) / self.sum_tree.get_total_val()
  
        return indices, prios, self[indices]

    def update_priorities(self, indices, priorities):
        """
        Update priorities in sum tree of replay buffer.
        """
        
        # add small offset to ensure that transitions with zero error can also be replayed
        # interpolate between greedy prioritization and uniform random sampling
        priorities = (priorities + 1e-10) ** self.alpha
        self.max_priority = max(self.max_priority, onp.max(priorities))
        self.min_priority = min(self.min_priority, onp.min(priorities))
        self.sum_tree.update_values(indices, priorities)
        
    def serializable(self):
        """
        Get pickable representation of Replay Buffer.
        """
        tree_size = self.sum_tree.get_capacity()
        tree_index = range(tree_size)
        lst_serialize = [self.max_priority,
                         self.min_priority,
                         self.alpha,
                         tree_size,
                         self.sum_tree.get_values(tree_index)]
        return super().serializable(), lst_serialize
    
    def load(self, lst_serializable):
        """
        Load pickable representation of Replay Buffer. Inverse function of serializable
        """
        super().load(lst_serializable[0])
        self.max_priority = lst_serializable[1][0]
        self.min_priority = lst_serializable[1][1]
        self.alpha = lst_serializable[1][2]
        capacity = lst_serializable[1][3]
        tree_index = range(capacity)
        self.sum_tree = SumTree(capacity)
        self.sum_tree.update_values(tree_index, lst_serializable[1][4])


