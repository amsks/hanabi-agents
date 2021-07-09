from typing import NamedTuple, List, Callable, Union
import gin


@gin.configurable
class PPOParams(NamedTuple):
    """Parameterization class for PPO agent"""

    experience_buffer_size: int = 2**19         # Size of Experience Buffer
    history_size: int = 1                       # size of history for sampling from buffer
    gamma: float = 0.99                         # discount factor
    eps_clip: float = 0.2                       # clip parameter for PPO
    K_epochs: float = 80                        # update policy for K epochs in one PPO update
    lr_actor: float  = 0.0003                   # learning rate for actor network
    lr_critic: float = 0.001                    # learning rate for critic network
    max_ep_len: int = 1000                      # maximum length of an episode
    update_timestep: int = max_ep_len * 4       # update policy every n timesteps
    random_seed : int = 0                       # set random seed if required (0 = no random seed)                
    
@gin.configurable
class AgentType(NamedTuple):
    type: str = 'PPO'
