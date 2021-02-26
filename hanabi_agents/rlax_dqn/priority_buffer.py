from functools import partial
import numpy as onp
import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, tree_unflatten, register_pytree_node_class

from collections import namedtuple

#  from .sum_tree import SumTree
from sum_tree import SumTreef as SumTree
from .experience_buffer import ExperienceBuffer

class PriorityBuffer(ExperienceBuffer):
    def __init__(self, observation_len: int, action_len: int, reward_len: int,
                 capacity: int, alpha: int = 0.6):
        super(PriorityBuffer, self).__init__(observation_len, action_len, reward_len, capacity)
        self.sum_tree = SumTree(capacity)
        self.max_priority = alpha
        self.min_priority = alpha
        self.alpha = alpha

    def add_transitions(self,
                        observation_tm1: onp.ndarray,
                        action_tm1: onp.ndarray,
                        reward_t: onp.ndarray,
                        observation_t: onp.ndarray,
                        legal_moves_t: onp.ndarray,
                        terminal_t: onp.ndarray):
        batch_size = len(observation_tm1)
        self.sum_tree.update_values(
            self.get_update_indices(batch_size),
            [self.max_priority for _ in range(batch_size)])
        super(PriorityBuffer, self).add_transitions(
            observation_tm1, action_tm1, reward_t, observation_t, legal_moves_t, terminal_t)

    def sample_batch(self, batch_size):
        keys = onp.linspace(1. / batch_size, 1, batch_size)
        #  keys = keys * jax.random.uniform(next(self.rng), shape=(batch_size,), dtype=onp.float32) / batch_size / 2

        keys -= onp.random.uniform(size=(batch_size,), high=1./batch_size)
        indices = self.sum_tree.get_indices(keys)
        prios = (onp.array(self.sum_tree.get_values(indices)) + 1e-10) / self.sum_tree.get_total_val()
        #  p_min = self.min_priority / self.sum_tree.get_total_val()
        #  max_weight = (p_min * self.size) ** (-beta)
        #  weights_is = ((prios * self.size) ** (-beta)) / max_weight
        return indices, prios, self[indices]

    def update_priorities(self, indices, priorities):
        priorities = (priorities + 1e-10) ** self.alpha
        self.max_priority = max(self.max_priority, onp.max(priorities))
        self.min_priority = min(self.min_priority, onp.min(priorities))
        self.sum_tree.update_values(indices, priorities)


    def change_size(self, new_size):

        def get_indices_old_obs(new_size):
            # buffer being extended
            if self.capacity < new_size:
                # buffer full
                if self.size == self.capacity:
                    return 0, self.capacity, True
                # buffer not filled yet
                else:
                    return 0, self.oldest_entry, True
            # buffer being shrinked
            else:
                # buffer full
                if self.size == self.capacity:
                    # buffer-tail can be copied unfragmented
                    if self.oldest_entry >= new_size:
                        return (self.oldest_entry - new_size), self.oldest_entry, True
                    # most recent buffer entries are fragmented between bufferend and beginning
                    else:
                        
                        return ((self.capacity - (new_size - self.oldest_entry)), self.capacity ), (0, self.oldest_entry), False
                # buffer not yet full
                else:
                    if self.oldest_entry >= new_size:
                        return (self.oldest_entry - new_size), self.oldest_entry, True
                    else:
                        return 0, self.oldest_entry, True


        start_ind, end_ind, unfragmented = get_indices_old_obs(new_size)
        
        def ind_list(start_ind, end_ind, unfragmented):
            if unfragmented:
                return list(range(start_ind, end_ind))
            else:
                print('list from {} to {} and from {} to {}'.format(start_ind[0], start_ind[1], end_ind[0], end_ind[1]))
                list_1 = list(range(start_ind[0], start_ind[1]))
                list_2 = list(range(end_ind[0], end_ind[1]))
                list_1.extend(list_2)
                return list_1

        sumtree_ind = ind_list(start_ind, end_ind, unfragmented)
        intermediate_sumtree = SumTree(new_size)

        prio_values = self.sum_tree.get_values(sumtree_ind)
        #new sum_tree indices
        new_indices = list(range(0, len(prio_values)))
        intermediate_sumtree.update_values(new_indices, prio_values)
        self.sum_tree = intermediate_sumtree
        
        super(PriorityBuffer, self).change_size(new_size)

    def get_info(self):
        values = self.sum_tree.get_values(list(range(0, self.capacity)))
        return values

    def restore_sumtree(self, weights_sumtree):
        sumT = SumTree(self.capacity)
        values = weights_sumtree
        sumT.update_values(list(range(0, self.capacity)), values)
        self.sum_tree = sumT



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

@register_pytree_node_class
class TreeNode:
    def __init__(self,
                 left=None,
                 right=None,
                 index=-1,
                 priority: float = 0.0,
                 parent=None,
                 #  is_leaf=False,
                 transition=None):
        self.index = index
        self.priority = priority
        self.left = left
        self.right = right
        self.parent = parent
        #  self.is_leaf = is_leaf
        self.transition = transition

    def __repr__(self):
        return f"<TreeNode(priority={self.priority}, index={self.index})>"

    def __eq__(self, other):
        return self.index == other.index and self.priority == other.priority and self.transition == other.transition
    
    def tree_flatten(self):
        return ((self.left, self.right), (self.index, self.priority, self.parent, self.transition))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, *aux_data)

@register_pytree_node_class
class SumTreeNode:
    def __init__(self,
                 priority: float = 0.0,
                 left=None,
                 right=None,
                 parent=None):
        self.priority = priority
        self.children = [left, right]
        #  self.left = left
        #  self.right = right
        self.parent = parent

    def __repr__(self):
        return f"<SumTreeNode(priority={self.priority})>"
    
    def tree_flatten(self):
        return (self.children, (self.priority, self.parent))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data[0], *children, aux_data[1])

@register_pytree_node_class
class SumTreeLeaf(SumTreeNode):
    def __init__(self, priority: float = 0.0, transition: object = None, parent: SumTreeNode = None):
        self.priority = priority
        self.transition = transition
        self.parent = parent
        self.index = 0

    def __repr__(self):
        return f"<SumTreeLeaf(index={self.index}, priority={self.priority}, transition={self.transition})>"

    def tree_flatten(self):
        return ((self.priority,), (self.transition, self.parent))

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], *aux_data)

#  class PriorityBuffer:
#
#      def __init__(self, capacity: int):
#          self.capacity = capacity
#          #  self.root = SumTreeNode()
#          self.root = TreeNode()
#          #  self.leafs = [SumTreeLeaf() for _ in range(capacity)]
#          self.leafs = []
#          tree_depth = int(onp.ceil(onp.log2(capacity)))
#          self.max_priority = 1.0
#          self.oldest_node = 0
#
#          def add_children(node, depth):
#              if depth >= tree_depth - 1:
#                  node.left = TreeNode(parent=node)
#                  node.right = TreeNode(parent=node)
#                  self.leafs.extend([node.left, node.right])
#                  #  node.children[0] = SumTreeLeaf(parent=node)
#                  #  node.children[1] = SumTreeLeaf(parent=node)
#                  #  node.left = SumTreeLeaf(parent=node)
#                  #  node.right = SumTreeLeaf(parent=node)
#                  #  self.leafs.extend(node.children)
#                  return
#
#              node.left = TreeNode(parent=node)
#              node.right = TreeNode(parent=node)
#              add_children(node.left, depth + 1)
#              add_children(node.right, depth + 1)
#              #  node.children[0] = SumTreeNode(parent=node)
#              #  node.children[1] = SumTreeNode(parent=node)
#              #  add_children(node.children[0], depth + 1)
#              #  add_children(node.children[1], depth + 1)
#
#          add_children(self.root, 0)
#
#          for leaf_idx, leaf in enumerate(self.leafs):
#              leaf.index = leaf_idx
#              #  leaf.is_leaf = True
#
#      def add_new_sample(self, transition):
#          self.set_transition(self.oldest_node, transition)
#          self.update_priority(self.oldest_node, self.max_priority)
#          self.oldest_node = (self.oldest_node + 1) % self.capacity
#
#      def set_transition(self, leaf_idx, transition):
#          self.leafs[leaf_idx].transition = transition
#
#      def __get_item__(self, leaf_idx):
#          return self.leafs[leaf_idx]
#
#      def get_transition(self, leaf_idx):
#          return self.leafs[leaf_idx].transition
#
#      def update_priority(self, leaf_idx, priority):
#          node = self.leafs[leaf_idx]
#          prio_diff = priority - node.priority
#          node.priority = priority
#          node = node.parent
#          while node is not None:
#              node.priority += prio_diff
#              node = node.parent
#
#      def sample(self, query_value=None):
#          query_value = query_value or jax.random.uniform(next(self.rng))
#          query_value *= self.root.priority
#          node = self.root
#          while node.index == -1:
#              if query_value < node.left.priority:
#                  node = node.left
#              else:
#                  query_value += node.left.priority
#                  node = node.right
#          return node.index

    #  @staticmethod
    #  @jax.jit
    #  def _sample(root_node, query_value):
    #
    #      def go_left(v):
    #          return v[0].left, v[1]
    #
    #      def go_right(v):
    #          val = v[1] - v[0].left.priority
    #          return v[0].right, val
    #
    #      def fork(v):
    #          return jax.lax.cond(
    #                  v[1] < v[0].left.priority,
    #                  v,
    #                  go_left,
    #                  v,
    #                  go_right
    #                  )
    #
    #      node = root_node
    #      while not node.is_leaf:
    #          node, query_value = fork((node, query_value))
    #          #  if query_value < node.children[0].priority:
    #          #      node = node.children[0]
    #          #  else:
    #          #      query_value -= node.children[0].priority
    #          #      node = node.children[1]
    #      return node.index

    #  def sample(self, query_value=None):
        #  query_value = query_value or jax.random.uniform(next(self.rng))
        #  query_value *= self.root.priority
        #  return self._sample(self.root, query_value)
