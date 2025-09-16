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
    """Convert environment observation (LazyFrames/ndarray) into
    a float32 PyTorch tensor in [0,1], with an added batch dimension."""
    # Ensure LazyFrames are converted properly via __array__ interface
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)


class DQNAgent(BaseAgent):
    """Deep Q-Network Agent supporting:
    - Double Q-learning (stable target estimation)
    - Optional Dueling network architecture (value + advantage streams).
    """

    def __init__(self, env: Env, config: Dict, device: torch.device):
        # Store environment, config, and device
        self.env = env
        self.config = config
        self.device = device

        # Online network (policy) and target network (slow-moving copy)
        self.network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network = DuelingDQN(
            env.observation_space, env.action_space, config["dueling"]
        ).to(device)
        self.target_network.load_state_dict(self.network.state_dict())

        # Optimizer and replay buffer for experience replay
        self.optimizer = optim.Adam(
            self.network.parameters(), lr=config["learning_rate"], eps=1e-4
        )
        self.replay_buffer = ReplayBuffer(
            config["replay_buffer_size"],
            env.observation_space.shape,
            env.action_space.n,
            device,
        )

        # Huber loss is more robust to outliers than MSE
        self.loss_fn = nn.HuberLoss(delta=config["huber_loss_delta"])
        self.param_update_count = 0     # Counts gradient updates (for target syncs)
        self.frame_idx = 0              # Tracks frame index for epsilon scheduling

    @property
    def q_network(self):
        """Expose q_network for compatibility with BaseAgent or analysis tools."""
        return self.network

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action with epsilon-greedy exploration.
        - With prob ε: random action
        - Otherwise: greedy action w.r.t. current Q-values
        """
        if training:
            epsilon = self._get_current_epsilon()
            if random.random() < epsilon:
                return self.env.action_space.sample()

        obs_tensor = _obs_to_tensor(obs, self.device)
        with torch.no_grad():
            q_values = self.network(obs_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def add_experience(self, obs, action, reward, next_obs, done):
        """Store a transition in the replay buffer."""
        self.replay_buffer.add(obs, action, reward, next_obs, done)

    def train(self) -> Dict[str, float]:
        """Perform a single training step on a minibatch from replay buffer.
        Implements Double DQN target if enabled in config.
        Returns training diagnostics (loss, mean Q-value).
        """
        # Skip update until replay buffer is large enough
        if len(self.replay_buffer) < self.config["batch_size"]:
            return {}
        self.param_update_count += 1

        # Sample minibatch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.config["batch_size"]
        )

        # Compute target Q-values
        with torch.no_grad():
            if self.config["double_q"]:
                # Double DQN: select actions via online net, evaluate via target net
                next_action_q_values = self.network(next_states)
                best_next_actions = next_action_q_values.argmax(dim=1, keepdim=True)
                next_q_values_target = self.target_network(next_states)
                next_q_value = next_q_values_target.gather(1, best_next_actions)
            else:
                # Vanilla DQN: max over target network Q-values
                next_q_values = self.target_network(next_states)
                next_q_value, _ = next_q_values.max(dim=1, keepdim=True)
            target_q_value = rewards + self.config["gamma"] * next_q_value * (1 - dones)
        
        # Current Q-values for taken actions
        current_q_values = self.network(states)
        current_q_value = current_q_values.gather(1, actions)

        # Compute Huber loss between target and predicted Q-values
        loss = self.loss_fn(current_q_value, target_q_value)
        
        # Backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.network.parameters(),
            self.config["gradient_clip_norm"]
        )
        self.optimizer.step()
        
        # Periodically sync target network with online network
        if self.param_update_count % self.config["target_update_freq"] == 0:
            self.target_network.load_state_dict(self.network.state_dict())
        
        return {
            "q_learning_loss": loss.item(),
            "mean_q_value": current_q_values.mean().item()
        }

    def _get_current_epsilon(self) -> float:
        """Linear decay schedule for epsilon.
        Starts from epsilon_start → epsilon_end over epsilon_decay_frames.
        """
        start = self.config["epsilon_start"]
        end = self.config["epsilon_end"]
        decay_frames = self.config["epsilon_decay_frames"]
        fraction = (self.frame_idx - self.config["learning_starts"]) / decay_frames
        fraction = max(0.0, min(1.0, fraction))
        return start + fraction * (end - start)

    def save(self, path: str):
        """Save online Q-network parameters to disk."""
        torch.save(self.network.state_dict(), path)


    def load(self, path: str):
        """Load parameters into online and target networks (synchronized)."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.network.load_state_dict(checkpoint)