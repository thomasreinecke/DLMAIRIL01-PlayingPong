# utils/replay_buffer.py
import numpy as np
import torch


class ReplayBuffer:
    """A simple FIFO (first-in, first-out) experience replay buffer for DQN agents.
    Stores transitions (s, a, r, s', done) and allows uniform random sampling for training.
    """

    def __init__(self, size: int, obs_shape: tuple, action_dim: int, device: torch.device):
        # Maximum number of transitions the buffer can hold
        self.max_size = size
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self.device = device

        # Pointer to the next insertion index and the number of filled slots
        self.ptr = 0
        self.current_size = 0

        # Preallocate numpy arrays for efficiency
        # States and next states are stored as uint8 (raw frames) to save memory
        self.states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((size, 1), dtype=np.int64)
        self.rewards = np.zeros((size, 1), dtype=np.float32)
        self.next_states = np.zeros((size, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros((size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        """Add a new transition to the buffer (overwrites oldest when full)."""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        # Circular pointer: wrap around when buffer is full
        self.ptr = (self.ptr + 1) % self.max_size
        # Increase size until reaching max capacity
        self.current_size = min(self.current_size + 1, self.max_size)

    def sample(self, batch_size: int):
        """Uniformly sample a batch of transitions for training."""
        indices = np.random.randint(0, self.current_size, size=batch_size)

        # Convert to float tensors (normalize states to [0,1] for training stability)
        states = torch.from_numpy(self.states[indices]).float().to(self.device) / 255.0
        actions = torch.from_numpy(self.actions[indices]).to(self.device)
        rewards = torch.from_numpy(self.rewards[indices]).to(self.device)
        next_states = torch.from_numpy(self.next_states[indices]).float().to(self.device) / 255.0
        dones = torch.from_numpy(self.dones[indices]).to(self.device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current number of stored transitions."""
        return self.current_size
