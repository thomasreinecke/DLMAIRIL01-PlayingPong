# agents/ppo_agent.py
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import Env

from agents.base_agent import BaseAgent
from models.ppo_model import PPOActorCritic
from utils.logger import Logger


def _obs_to_tensor(obs, device, add_batch: bool):
    """Handle LazyFrames/ndarray → float32 tensor in [0,1]."""
    arr = np.asarray(obs, dtype=np.float32)  # handles LazyFrames
    arr /= 255.0
    t = torch.from_numpy(arr).to(device)
    if add_batch:
        t = t.unsqueeze(0)  # (1, C, H, W)
    return t


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization Agent (single-env setup)."""

    def __init__(self, env: Env, config: Dict, logger: Logger, device: torch.device):
        self.env = env
        self.config = config
        self.logger = logger
        self.device = device

        self.num_envs = env.num_envs if hasattr(env, "num_envs") else 1
        self.rollout_len = config["rollout_len"]
        self.batch_size = self.num_envs * self.rollout_len
        self.minibatch_size = self.batch_size // config["num_minibatches"]

        self.network = PPOActorCritic(env.observation_space, env.action_space).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config["learning_rate"], eps=1e-5)

        # Storage for rollouts (float32 normalized)
        obs_shape = env.observation_space.shape
        self.obs = torch.zeros((self.rollout_len, self.num_envs) + obs_shape, dtype=torch.float32, device=device)
        self.actions = torch.zeros((self.rollout_len, self.num_envs) + env.action_space.shape, device=device)
        self.log_probs = torch.zeros((self.rollout_len, self.num_envs), device=device)
        self.rewards = torch.zeros((self.rollout_len, self.num_envs), device=device)
        self.dones = torch.zeros((self.rollout_len, self.num_envs), device=device)
        self.values = torch.zeros((self.rollout_len, self.num_envs), device=device)

        self.step = 0

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action from the policy (used during evaluation)."""
        obs_tensor = _obs_to_tensor(obs, self.device, add_batch=True)
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor)
        return int(action.squeeze().item())

    def collect_rollout(self, next_obs, next_done):
        """Collect one step of experience."""
        self.obs[self.step] = _obs_to_tensor(next_obs, self.device, add_batch=False)
        self.dones[self.step] = torch.as_tensor(next_done, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            action, log_prob, _, value = self.network.get_action_and_value(self.obs[self.step])
            self.values[self.step] = value.flatten()

        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        return action.cpu().numpy()

    def set_reward(self, reward):
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device=self.device).view(-1)

    def finish_step(self):
        self.step = (self.step + 1) % self.rollout_len

    def should_train(self) -> bool:
        return self.step == 0

    def train(self, next_obs, next_done) -> Dict[str, float]:
        """Perform a PPO update over the collected rollout."""
        # --- GAE advantages ---
        with torch.no_grad():
            next_value = self.network.get_value(_obs_to_tensor(next_obs, self.device, add_batch=False)).reshape(1, -1)
            advantages = torch.zeros_like(self.rewards, device=self.device)
            last_gae_lam = 0.0
            for t in reversed(range(self.rollout_len)):
                if t == self.rollout_len - 1:
                    next_non_terminal = 1.0 - torch.as_tensor(next_done, dtype=torch.float32, device=self.device)
                    next_values = next_value
                else:
                    next_non_terminal = 1.0 - self.dones[t + 1]
                    next_values = self.values[t + 1]
                delta = self.rewards[t] + self.config["gamma"] * next_values * next_non_terminal - self.values[t]
                advantages[t] = last_gae_lam = delta + self.config["gamma"] * self.config["gae_lambda"] * next_non_terminal * last_gae_lam
            returns = advantages + self.values

        # --- Flatten batch ---
        b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
        b_log_probs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)

        # --- Optimize ---
        b_inds = np.arange(self.batch_size)
        for epoch in range(self.config["num_update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                log_ratio = new_log_prob - b_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)

                mb_advantages = b_advantages[mb_inds]
                if self.config["normalize_advantages"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[mb_inds]) ** 2).mean()

                # Entropy loss
                entropy_loss = entropy.mean()

                # Total loss
                loss = pg_loss - self.config["entropy_coef"] * entropy_loss + v_loss * self.config["value_loss_coef"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config["gradient_clip_norm"])
                self.optimizer.step()

        return {
            "loss": loss.item(),
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
        }

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))
