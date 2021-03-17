from typing import NamedTuple, List, Callable, Union
import gin

@gin.configurable
class RlaxRainbowParams(NamedTuple):
    """Parameterization class for Rlax-based Rainbow DQN agent."""

    train_batch_size: int = 256
    target_update_period: int = 500
    discount: float = 0.99
    epsilon: Union[Callable[[int], float], float] = lambda x: 0.0
    learning_rate: float = 6.25e-5
    layers: List[int] = [512, 512]
    use_double_q: bool = True
    use_priority: bool = False
    experience_buffer_size: int = 2**15
    seed: int = 1234
    n_atoms: int = 31
    atom_vmax: int = 25
    beta_is: Union[Callable[[int], float], float] = lambda x: 0.4
    priority_w: float = 0.6
    history_size: int = 1
    
@gin.configurable
class RewardShapingParams(NamedTuple):
    
    # conservative agent
    use_reward_shaping: bool = True
    min_play_probability: float = 0
    w_play_probability: float = 0
    penalty_last_of_kind: float = 0

@gin.configurable
class PBTParams(NamedTuple):

    """Parametrization for a population of dqn/rainbow agents

    PBT_parameters:
        population_size -- Number of atomic agents.
        gin_files -- Gin-File with RL-agent config
        agent_config_path --
        lr_factor -- Factor by which the lr gets changes during evolutionary step
        lr_start_var -- For first initialization of lr: lr is sampled from interval [gin_file-lr / lr_start_var : gin_file-lr * lr_start_var]
        buffersize -- Initial buffersize for each atomic agent is sampled from this list.
        buffer_factor -- On evolutionary step, buffersizes are shrinked/extended by this factor.
        discard_perc -- Percentage of atomic agents to discard of total population during evolutionary steps.
    """

    population_size: int = 20
    discard_percent: float = 0.8
    individual_reward_shaping: bool = False
    life_span: int = 1
    generations: int = 3500
    saver_threshold: float = 13
    pool_path: str = 'pool'
    use_db: bool = True
    db_path: str = 'obs.db'
    w_diversity: float = 10
    obs_no: int = 10000
    n_mean: int = 5


    change_alpha: bool = True
    factor_alpha: float = 1.05
    alpha_min: float = 0.3
    alpha_max: float = 1
    alpha_sample_size: int = 10

    change_buffersize: bool = True
    buffersize_start_factor: int = 4
    buffersize_factor: int = 2

    change_learning_rate: bool = True
    lr_factor: float = 0.2
    lr_min: float = 1e-3
    lr_max: float = 1e-6
    lr_sample_size: int = 10

    change_min_play_probability: bool = False
    change_w_play_probability: bool = False
    min_play_probability_pbt: float = 0.05
    w_play_probability_pbt: float = 0.05

    change_penalty_last_of_kind: bool = False
    penalty_last_of_kind_pbt: float = 0.1


