# agents/base_agent.py
import abc
from typing import Any, Dict


class BaseAgent(abc.ABC):
    """Abstract base class for a reinforcement learning agent."""

    @abc.abstractmethod
    def act(self, obs: Any, training: bool = True) -> Any:
        """Select an action based on the observation."""
        raise NotImplementedError

    @abc.abstractmethod
    def train(self, *args, **kwargs) -> Dict[str, float]:
        """Perform a single training step."""
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, path: str):
        """Save the agent's model."""
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, path: str):
        """Load the agent's model."""
        raise NotImplementedError