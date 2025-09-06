# models/ppo_model.py
import numpy as np
import torch
import torch.nn as nn
from gymnasium.spaces import Space
from torch.distributions.categorical import Categorical


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActorCritic(nn.Module):
    """Shared-parameter Actor-Critic architecture for PPO (Atari-style CNN)."""

    def __init__(self, obs_space: Space, action_space: Space):
        super().__init__()
        self.action_dim = action_space.n

        # Detect channel order; assume channels-last if last dim looks like channels
        shape = obs_space.shape
        self.channels_last = (len(shape) == 3) and (shape[-1] in (1, 3, 4))
        if self.channels_last:
            C, H, W = shape[-1], shape[0], shape[1]
        else:
            C, H, W = shape[0], shape[1], shape[2]

        # CNN backbone
        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(C, 32, kernel_size=8, stride=4), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1), std=np.sqrt(2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened feature size
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            feat_size = int(np.prod(self.cnn(dummy).shape[1:]))

        # Actor & Critic heads
        self.actor_head = nn.Sequential(
            layer_init(nn.Linear(feat_size, 512), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(512, self.action_dim), std=0.01),
        )
        self.critic_head = nn.Sequential(
            layer_init(nn.Linear(feat_size, 512), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 1), std=1.0),
        )

    # ----- helpers -----
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Permute (N,H,W,C) → (N,C,H,W) if channels are last."""
        if self.channels_last:
            return x.permute(0, 3, 1, 2)
        return x

    # ----- API -----
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(x)
        return self.critic_head(self.cnn(x))

    def get_policy_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_nchw(x)
        return self.actor_head(self.cnn(x))

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample/evaluate action; return (action, log_prob, entropy, value)."""
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
