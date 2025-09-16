# models/ppo_model.py
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Utility for orthogonal weight initialization with controlled std/bias.
    - Conv/linear layers initialized with orthogonal matrices
    - Bias set to constant
    """
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActorCritic(nn.Module):
    """Shared-parameter Actor-Critic network used by PPO.
    - CNN backbone encodes observations (Atari-style conv stack).
    - Actor head outputs action logits (policy).
    - Critic head outputs state-value estimate.
    """

    def __init__(self, obs_space: Space, action_space: Space):
        super().__init__()
        self.action_dim = action_space.n

        # Handle observation layout: support (H, W, C) or (C, H, W)
        shape = obs_space.shape
        self.channels_last = (len(shape) == 3) and (shape[-1] in (1, 3, 4))
        if self.channels_last:
            C, H, W = shape[-1], shape[0], shape[1]
        else:
            C, H, W = shape[0], shape[1], shape[2]

        # CNN feature extractor (same backbone for actor & critic)
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=8, stride=4), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), std=np.sqrt(2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically determine feature size from CNN output
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            feat_size = int(np.prod(self.cnn(dummy).shape[1:]))

        # Actor head → outputs logits for categorical policy
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(feat_size, 512), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(512, self.action_dim), std=0.01),  # small init for logits
        )
        # Critic head → outputs scalar state-value
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(feat_size, 512), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    # ----- helpers -----
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input has shape (N, C, H, W).
        Permute if input is (N, H, W, C)."""
        if self.channels_last:
            return x.permute(0, 3, 1, 2)
        return x

    # ----- API -----
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """Return critic’s state-value estimate V(s)."""
        x = self._to_nchw(x)
        return self.critic_head(self.cnn(x))

    def get_policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw action logits from the actor head (before softmax)."""
        x = self._to_nchw(x)
        return self.actor_head(self.cnn(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample or evaluate actions and compute value estimates.
        Returns: (action, log_prob, entropy, value)
        - If action is None: sample from policy distribution.
        - If action is provided: evaluate its log-prob + value.
        """
        x = self._to_nchw(x)
        feats = self.cnn(x)
        logits = self.actor_head(feats)
        dist = Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic_head(feats)
        return action, log_prob, entropy, value
