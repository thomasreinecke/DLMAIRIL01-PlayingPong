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

        # Detect channel order; assume channels-last if last dim looks like channels
        shape = obs_space.shape
        self.channels_last = shape[-1] in (1, 3, 4) and len(shape) == 3
        if self.channels_last:
            C, H, W = shape[-1], shape[0], shape[1]
        else:
            C, H, W = shape[0], shape[1], shape[2]

        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size with correct layout
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            feature_size = int(np.prod(self.cnn(dummy_input).shape[1:]))

        self.actor_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.action_dim),
        )

        self.critic_head = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Permutes an (N, H, W, C) tensor to (N, C, H, W) if channels are last."""
        if self.channels_last:
            return x.permute(0, 3, 1, 2)
        return x

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Get state value from the critic head."""
        x = self._to_nchw(x)
        return self.critic_head(self.cnn(x))

    def get_policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get policy logits from the actor head."""
        x = self._to_nchw(x)
        return self.actor_head(self.cnn(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None):
        """Get action, log probability, entropy, and state value."""
        x = self._to_nchw(x)
        features = self.cnn(x)
        logits = self.actor_head(features)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        log_prob = probs.log_prob(action)
        entropy = probs.entropy()
        value = self.critic_head(features)

        return action, log_prob, entropy, value