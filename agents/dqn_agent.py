# agents/dqn_agent.py
import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env

from agents.base_agent import BaseAgent
from models.dqn_model import DuelingDQN
from utils.logger import Logger
from utils.replay_buffer import ReplayBuffer


def _obs_to_tensor(obs, device):
    """Handle LazyFrames/ndarray → float32 tensor in [0,1], with batch dim."""
    arr = np.asarray(obs, dtype=np.float32)  # handles LazyFrames via __array__
    arr /= 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)  # (1, C, H, W)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent with Double Q-Learning and optional Dueling architecture."""

    def __init__(self, env: Env, config: Dict, logger: Logger, device: torch.device):
        self.env = env
        self.config = config
        self.logger = logger
        self.device = device

        self.q_network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=config["learning_rate"]
        )
        self.replay_buffer = ReplayBuffer(
            config["replay_buffer_size"],
            env.observation_space.shape,
            env.action_space.n,
            device,
        )
        self.loss_fn = nn.HuberLoss(delta=config["huber_loss_delta"])

        self.step_count = 0

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action using an epsilon-greedy policy."""
        if training:
            self.step_count += 1
            epsilon = self._get_current_epsilon()
            if random.random() < epsilon:
                return self.env.action_space.sample()

        obs_tensor = _obs_to_tensor(obs, self.device)
        with torch.no_grad():
            q_values = self.q_network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def add_experience(self, obs, action, reward, next_obs, done):
        """Add a transition to the replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def train(self) -> Dict[str, float]:
        """Perform one step of training."""
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {}

        # Sample a minibatch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config["batch_size"]
        )

        # Compute the target Q-value
        with torch.no_grad():
            if self.config["double_q"]:
                # Double DQN: Select action with online network, evaluate with target network
                next_action_q_values = self.q_network(next_states)
                best_next_actions = next_action_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = self.target_network(next_states)
                next_q_value = next_q_values_target.gather(1, best_next_actions)
            else:
                # Standard DQN: Max Q-value from target network
                next_q_values = self.target_network(next_states)
                next_q_value, _ = next_q_values.max(dim=1, keepdim=True)

            target_q_value = rewards + self.config["gamma"] * next_q_value * (1 - dones)

        # Get current Q-value estimates
        current_q_values = self.q_network(states)
        current_q_value = current_q_values.gather(1, actions)

        # Compute loss
        loss = self.loss_fn(current_q_value, target_q_value)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.config["gradient_clip_norm"])
        self.optimizer.step()

        # Periodically update the target network
        if self.step_count % self.config["target_update_freq"] == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item()}

    def _get_current_epsilon(self) -> float:
        """Calculate epsilon based on the current frame count."""
        start = self.config["epsilon_start"]
        end = self.config["epsilon_end"]
        decay_frames = self.config["epsilon_decay_frames"]
        fraction = (self.step_count - self.config["learning_starts"]) / decay_frames
        fraction = max(0.0, min(1.0, fraction))  # clamp to [0, 1]
        return start + fraction * (end - start)

    def save(self, path: str):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str):
        self.q_network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.q_network.state_dict())
