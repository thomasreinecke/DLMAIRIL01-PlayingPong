# models/ppo_model.py
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Space
from torch.distributions.categorical import Categorical


class PPOActorCritic(nn.Module):
    """Shared-parameter Actor-Critic architecture for PPO."""

    def __init__(self, obs_space: Space, action_space: Space):
        super().__init__()
        self.action_dim = action_space.n

        # CNN backbone for feature extraction (shared)
        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened feature size
        with torch.no_grad():
            dummy_input = torch.as_tensor(obs_space.sample()[None]).float()
            feature_size = np.prod(self.cnn(dummy_input).shape)

        # Actor head for policy
        self.actor_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim),
        )

        # Critic head for value function
        self.critic_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value from the critic head."""
        return self.critic_head(self.cnn(x / 255.0))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """Get action, log probability, entropy, and state value."""
        features = self.cnn(x / 255.0)
        logits = self.actor_head(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic_head(features)

        return action, log_prob, entropy, value