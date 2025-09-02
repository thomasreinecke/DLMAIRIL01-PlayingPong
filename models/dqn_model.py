# models/dqn_model.py
import torch
import torch.nn as nn
import numpy as np
from gymnasium.spaces import Space


class DuelingDQN(nn.Module):
    """Dueling Deep Q-Network architecture."""

    def __init__(self, obs_space: Space, action_space: Space, dueling: bool = True):
        super().__init__()
        self.action_dim = action_space.n
        self.dueling = dueling

        # CNN backbone for feature extraction
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

        # Dueling network streams
        if self.dueling:
            self.advantage_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )
            self.value_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )
        else:
            # Standard DQN output layer
            self.q_stream = nn.Sequential(
                nn.Linear(feature_size, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute Q-values."""
        features = self.cnn(x)
        if self.dueling:
            value = self.value_stream(features)
            advantages = self.advantage_stream(features)
            # Combine value and advantages for Q-values
            q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        else:
            q_values = self.q_stream(features)
        return q_values