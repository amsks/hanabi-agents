import numpy as np
import pickle

from sum_tree import SumTreef as SumTree
from .experience_buffer import ExperienceBuffer

class PriorityBuffer(ExperienceBuffer):
    
    def __init__(self, capacity: int, obs_len: int, n_network: int, alpha: int = 0.6):

        super(PriorityBuffer, self).__init__(capacity, obs_len, n_network)
        self.td_buf = np.empty((n_network, self.capacity, 1), dtype=np.float64)
        
        self.sum_tree = [SumTree(capacity) for _ in range(n_network)]
        self.max_priority = [alpha for _ in range(n_network)]
        self.min_priority = [alpha for _ in range(n_network)]
        self.alpha = alpha
        
    def add(self, 
            observation_tm1: np.ndarray,
            action_tm1: np.ndarray,
            reward_t: np.ndarray,
            observation_t: np.ndarray,
            terminal_t: np.ndarray):
        
        batch_size = observation_tm1.shape[1] 
        update_indices = self.get_update_indices(batch_size)
        self.td_buf[:, update_indices, :] = 0
        
        for i in range(self.n_network):
            self.sum_tree[i].update_values(
                update_indices,
                [self.max_priority[i] for _ in range(batch_size)]
            )

        super(PriorityBuffer, self).add(
            observation_tm1, action_tm1, reward_t, observation_t, terminal_t
        )
        
    def sample(self, batch_size):
        
        step_size = 1 / batch_size
        keys = np.broadcast_to(np.linspace(step_size, 1, batch_size), (self.n_network, batch_size))
        keys = keys - np.random.uniform(size=(self.n_network, batch_size), high=step_size)

        indices = np.array([tree.get_indices(keys[i]) for i, tree in enumerate(self.sum_tree)])
        tree_root_value = 1 / np.array([tree.get_total_val() for tree in self.sum_tree])
        prios = np.array([tree.get_values(indices[i]) for i, tree in enumerate(self.sum_tree)])
        prios = (prios + 1e-10) * tree_root_value[:, np.newaxis]
        
        return self[indices], indices, prios

    def update_priorities(self, indices, tds):
        
        priorities = (tds + 1e-10) ** self.alpha
        #priorities = np.sqrt((priorities + 1e-10)) # faster version for alpha=0.5
        self.max_priority = np.maximum(self.max_priority, np.amax(priorities, axis=1))
        self.min_priority = np.minimum(self.min_priority, np.amin(priorities, axis=1))
        
        for i in range(self.n_network):
            self.td_buf[i, indices[i], :] = tds
            self.sum_tree[i].update_values(indices[i], priorities[i])
            
    def get_tds(self, indices): 
        return self.td_buf[:,indices, :]
        
    def serializable(self):
        
        tree_index = range(self.capacity)
        lst_serialize = [self.max_priority,
                         self.min_priority,
                         self.alpha,
                         self.capacity,
                         [tree.get_values(tree_index) for tree in self.sum_tree]]
        return super().serializable(), lst_serialize
    
    def load(self, lst_serializable):
        super().load(lst_serializable[0])
        self.max_priority = lst_serializable[1][0]
        self.min_priority = lst_serializable[1][1]
        self.alpha = lst_serializable[1][2]
        self.capacity = lst_serializable[1][3]
        tree_index = range(capacity)
        self.sum_tree = [SumTree(capacity) for _ in range(self.n_network)]
        for i in range(self.n_network):
            self.sum_tree[i].update_values(tree_index, lst_serializable[1][4][i])
        
#  @register_pytree_node_class
#  class Transition:
#      def __init__(self, obs_tm1, action_tm1, reward_tm1, obs_t):
#          self.obs_tm1 = obs_tm1
#          self.action_tm1 = action_tm1
#          self.reward_tm1 = reward_tm1
#          self.obs_t = obs_t
#
#      def tree_flatten(self):
#          return ((self.obs_tm1, self.action_tm1, self.reward_tm1, self.obs_t), None)
#
#      @classmethod
#      def tree_unflatten(cls, aux_data, children):
#          return cls(*children)

# @register_pytree_node_class
# class TreeNode:
#     def __init__(self,
#                  left=None,
#                  right=None,
#                  index=-1,
#                  priority: float = 0.0,
#                  parent=None,
#                  #  is_leaf=False,
#                  transition=None):
#         self.index = index
#         self.priority = priority
#         self.left = left
#         self.right = right
#         self.parent = parent
#         #  self.is_leaf = is_leaf
#         self.transition = transition
# 
#     def __repr__(self):
#         return f"<TreeNode(priority={self.priority}, index={self.index})>"
# 
#     def __eq__(self, other):
#         return self.index == other.index and self.priority == other.priority and self.transition == other.transition
#     
#     def tree_flatten(self):
#         return ((self.left, self.right), (self.index, self.priority, self.parent, self.transition))
# 
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(*children, *aux_data)
# 
# @register_pytree_node_class
# class SumTreeNode:
#     def __init__(self,
#                  priority: float = 0.0,
#                  left=None,
#                  right=None,
#                  parent=None):
#         self.priority = priority
#         self.children = [left, right]
#         #  self.left = left
#         #  self.right = right
#         self.parent = parent
# 
#     def __repr__(self):
#         return f"<SumTreeNode(priority={self.priority})>"
#     
#     def tree_flatten(self):
#         return (self.children, (self.priority, self.parent))
# 
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(aux_data[0], *children, aux_data[1])
# 
# @register_pytree_node_class
# class SumTreeLeaf(SumTreeNode):
#     def __init__(self, priority: float = 0.0, transition: object = None, parent: SumTreeNode = None):
#         self.priority = priority
#         self.transition = transition
#         self.parent = parent
#         self.index = 0
# 
#     def __repr__(self):
#         return f"<SumTreeLeaf(index={self.index}, priority={self.priority}, transition={self.transition})>"
# 
#     def tree_flatten(self):
#         return ((self.priority,), (self.transition, self.parent))
# 
#     @classmethod
#     def tree_unflatten(cls, aux_data, children):
#         return cls(children[0], *aux_data)
# 
# #  class PriorityBuffer:
# #
# #      def __init__(self, capacity: int):
# #          self.capacity = capacity
# #          #  self.root = SumTreeNode()
# #          self.root = TreeNode()
# #          #  self.leafs = [SumTreeLeaf() for _ in range(capacity)]
# #          self.leafs = []
# #          tree_depth = int(onp.ceil(onp.log2(capacity)))
# #          self.max_priority = 1.0
# #          self.oldest_node = 0
# #
# #          def add_children(node, depth):
# #              if depth >= tree_depth - 1:
# #                  node.left = TreeNode(parent=node)
# #                  node.right = TreeNode(parent=node)
# #                  self.leafs.extend([node.left, node.right])
# #                  #  node.children[0] = SumTreeLeaf(parent=node)
# #                  #  node.children[1] = SumTreeLeaf(parent=node)
# #                  #  node.left = SumTreeLeaf(parent=node)
# #                  #  node.right = SumTreeLeaf(parent=node)
# #                  #  self.leafs.extend(node.children)
# #                  return
# #
# #              node.left = TreeNode(parent=node)
# #              node.right = TreeNode(parent=node)
# #              add_children(node.left, depth + 1)
# #              add_children(node.right, depth + 1)
# #              #  node.children[0] = SumTreeNode(parent=node)
# #              #  node.children[1] = SumTreeNode(parent=node)
# #              #  add_children(node.children[0], depth + 1)
# #              #  add_children(node.children[1], depth + 1)
# #
# #          add_children(self.root, 0)
# #
# #          for leaf_idx, leaf in enumerate(self.leafs):
# #              leaf.index = leaf_idx
# #              #  leaf.is_leaf = True
# #
# #      def add_new_sample(self, transition):
# #          self.set_transition(self.oldest_node, transition)
# #          self.update_priority(self.oldest_node, self.max_priority)
# #          self.oldest_node = (self.oldest_node + 1) % self.capacity
# #
# #      def set_transition(self, leaf_idx, transition):
# #          self.leafs[leaf_idx].transition = transition
# #
# #      def __get_item__(self, leaf_idx):
# #          return self.leafs[leaf_idx]
# #
# #      def get_transition(self, leaf_idx):
# #          return self.leafs[leaf_idx].transition
# #
# #      def update_priority(self, leaf_idx, priority):
# #          node = self.leafs[leaf_idx]
# #          prio_diff = priority - node.priority
# #          node.priority = priority
# #          node = node.parent
# #          while node is not None:
# #              node.priority += prio_diff
# #              node = node.parent
# #
# #      def sample(self, query_value=None):
# #          query_value = query_value or jax.random.uniform(next(self.rng))
# #          query_value *= self.root.priority
# #          node = self.root
# #          while node.index == -1:
# #              if query_value < node.left.priority:
# #                  node = node.left
# #              else:
# #                  query_value += node.left.priority
# #                  node = node.right
# #          return node.index
# 
#     #  @staticmethod
#     #  @jax.jit
#     #  def _sample(root_node, query_value):
#     #
#     #      def go_left(v):
#     #          return v[0].left, v[1]
#     #
#     #      def go_right(v):
#     #          val = v[1] - v[0].left.priority
#     #          return v[0].right, val
#     #
#     #      def fork(v):
#     #          return jax.lax.cond(
#     #                  v[1] < v[0].left.priority,
#     #                  v,
#     #                  go_left,
#     #                  v,
#     #                  go_right
#     #                  )
#     #
#     #      node = root_node
#     #      while not node.is_leaf:
#     #          node, query_value = fork((node, query_value))
#     #          #  if query_value < node.children[0].priority:
#     #          #      node = node.children[0]
#     #          #  else:
#     #          #      query_value -= node.children[0].priority
#     #          #      node = node.children[1]
#     #      return node.index
# 
#     #  def sample(self, query_value=None):
#         #  query_value = query_value or jax.random.uniform(next(self.rng))
#         #  query_value *= self.root.priority
#         #  return self._sample(self.root, query_value)
