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
from utils.replay_buffer import ReplayBuffer


def _obs_to_tensor(obs, device):
    """Handle LazyFrames/ndarray → float32 tensor in [0,1], with batch dim."""
    # handles LazyFrames via __array__
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent with Double Q-Learning and optional Dueling architecture."""

    def __init__(self, env: Env, config: Dict, device: torch.device):
        self.env = env
        self.config = config
        self.device = device

        self.network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config["learning_rate"], eps=1e-4
        )
        self.replay_buffer = ReplayBuffer(
            config["replay_buffer_size"],
            env.observation_space.shape,
            env.action_space.n,
            device,
        )
        self.loss_fn = nn.HuberLoss(delta=config["huber_loss_delta"])
        self.param_update_count = 0
        self.frame_idx = 0 # To be updated by training loop for epsilon

    @property
    def q_network(self): # Keep property for backward compatibility if needed elsewhere
        return self.network

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action using an epsilon-greedy policy."""
        if training:
            epsilon = self._get_current_epsilon()
            if random.random() < epsilon:
                return self.env.action_space.sample()

        obs_tensor = _obs_to_tensor(obs, self.device)
        with torch.no_grad():
            q_values = self.network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def add_experience(self, obs, action, reward, next_obs, done):
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def train(self) -> Dict[str, float]:
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {}
        self.param_update_count += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config["batch_size"]
        )
        with torch.no_grad():
            if self.config["double_q"]:
                next_action_q_values = self.network(next_states)
                best_next_actions = next_action_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = self.target_network(next_states)
                next_q_value = next_q_values_target.gather(1, best_next_actions)
            else:
                next_q_values = self.target_network(next_states)
                next_q_value, _ = next_q_values.max(dim=1, keepdim=True)
            target_q_value = rewards + self.config["gamma"] * next_q_value * (1 - dones)
        
        current_q_values = self.network(states)
        current_q_value = current_q_values.gather(1, actions)
        loss = self.loss_fn(current_q_value, target_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.network.parameters(), self.config["gradient_clip_norm"])
        self.optimizer.step()
        
        if self.param_update_count % self.config["target_update_freq"] == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        return {
            "q_learning_loss": loss.item(),
            "mean_q_value": current_q_values.mean().item()
        }

    def _get_current_epsilon(self) -> float:
        start = self.config["epsilon_start"]
        end = self.config["epsilon_end"]
        decay_frames = self.config["epsilon_decay_frames"]
        fraction = (self.frame_idx - self.config["learning_starts"]) / decay_frames
        fraction = max(0.0, min(1.0, fraction))
        return start + fraction * (end - start)

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))
        self.target_network.load_state_dict(self.network.state_dict())