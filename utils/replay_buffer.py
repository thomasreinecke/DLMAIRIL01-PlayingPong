# utils/replay_buffer.py
import numpy as np
import torch


class ReplayBuffer:
    """A simple FIFO experience replay buffer for DQN agents."""

    def __init__(self, size: int, obs_shape: tuple, action_dim: int, device: torch.device):
        self.max_size = size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        self.ptr = 0
        self.current_size = 0

        self.states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        """Sample a batch of transitions from the buffer."""
        indices = np.random.randint(0, self.current_size, size=batch_size)

        states = torch.from_numpy(self.states[indices]).float().to(self.device) / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device) / 255.0
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.current_size