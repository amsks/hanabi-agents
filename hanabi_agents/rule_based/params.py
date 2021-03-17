from typing import NamedTuple, List, Callable, Union
import gin
from hanabi_agents.rule_based.predefined_rules import piers_rules

@gin.configurable
class RulebasedParams(NamedTuple):
    """Parameterization class for Rule-based agent."""

    ruleset: str = piers_rules