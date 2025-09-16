# models/dqn_model.py
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Space


class DuelingDQN(nn.Module):
    """Deep Q-Network (with optional Dueling architecture).
    - CNN feature extractor (Atari-style conv stack).
    - Either: single Q-stream OR separate value/advantage streams.
    """

    def __init__(self, obs_space: Space, action_space: Space, dueling: bool = True):
        super().__init__()
        self.action_dim = action_space.n
        self.dueling = dueling

        # Handle observation layout: support (H, W, C) or (C, H, W)
        shape = obs_space.shape
        self.channels_last = shape[-1] in (1, 3, 4) and len(shape) == 3
        if self.channels_last:
            C, H, W = shape[-1], shape[0], shape[1]
        else:
            C, H, W = shape[0], shape[1], shape[2]

        # Convolutional feature extractor (standard DQN backbone)
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Dynamically compute flattened feature size for FC layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, C, H, W)
            feature_size = int(np.prod(self.cnn(dummy_input).shape[1:]))

        if self.dueling:
            # Separate advantage stream: estimates relative action preferences
            self.advantage_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )
            # Separate value stream: estimates baseline state value
            self.value_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        else:
            # Vanilla DQN: single stream for Q-values
            self.q_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )
    
    def _to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        """Ensure input has shape (N, C, H, W).
        Permute if input is (N, H, W, C)."""
        if self.channels_last:
            return x.permute(0, 3, 1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values for each action."""
        x = self._to_nchw(x)           # Adjust layout if needed
        features = self.cnn(x)         # Extract visual features
        if self.dueling:
            value = self.value_stream(features)              # scalar state-value
            advantages = self.advantage_stream(features)     # per-action advantages
            # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_stream(features)               # Direct Q-value output
        return q_values
