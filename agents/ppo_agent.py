# agents/ppo_agent.py
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn  # <-- THIS IS THE FIX
from gymnasium import Env
from agents.base_agent import BaseAgent
from models.ppo_model import PPOActorCritic

def _obs_to_tensor(obs, device):
    arr = np.asarray(obs, dtype=np.float32)
    return torch.from_numpy(arr).unsqueeze(0).to(device)

class PPOAgent(BaseAgent):
    def __init__(self, env: Env, config: Dict, device: torch.device):
        self.env = env
        self.config = config
        self.device = device
        self.network = PPOActorCritic(env.observation_space, env.action_space).to(device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=config["learning_rate"], eps=1e-5)
        
        self.num_envs = 1
        self.rollout_len = config["rollout_len"]
        self.batch_size = self.num_envs * self.rollout_len
        self.minibatch_size = self.batch_size // config["num_minibatches"]
        
        obs_shape = env.observation_space.shape
        self.obs = torch.zeros((self.rollout_len, self.num_envs) + obs_shape, dtype=torch.uint8, device="cpu")
        self.actions = torch.zeros((self.rollout_len, self.num_envs) + env.action_space.shape, device="cpu")
        self.log_probs = torch.zeros((self.rollout_len, self.num_envs), device="cpu")
        self.rewards = torch.zeros((self.rollout_len, self.num_envs), device="cpu")
        self.dones = torch.zeros((self.rollout_len, self.num_envs), device="cpu")
        self.values = torch.zeros((self.rollout_len, self.num_envs), device="cpu")
        self.step = 0

    def act(self, obs: np.ndarray, training: bool = True) -> int:
        obs_tensor = _obs_to_tensor(obs, self.device)
        with torch.no_grad():
            action, _, _, _ = self.network.get_action_and_value(obs_tensor)
        return int(action.squeeze().item())

    def collect_rollout(self, next_obs, next_done):
        """Collect one step of experience."""
        # Store raw uint8 obs to save memory, convert to tensor on CPU
        self.obs[self.step] = torch.as_tensor(np.asarray(next_obs), device="cpu")
        self.dones[self.step] = torch.as_tensor(next_done, device="cpu")
        with torch.no_grad():
            # Pass obs to network on correct device
            obs_tensor = self.obs[self.step].to(self.device).float()
            action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            self.values[self.step] = value.flatten().cpu()
        self.actions[self.step] = action.cpu()
        self.log_probs[self.step] = log_prob.cpu()
        return int(action.item())

    def set_reward(self, reward):
        self.rewards[self.step] = torch.as_tensor(reward, device="cpu").view(-1)

    def finish_step(self):
        self.step = (self.step + 1) % self.rollout_len

    def should_train(self) -> bool:
        return self.step == 0

    def train(self, next_obs, next_done) -> Dict[str, float]:
        with torch.no_grad():
            next_obs_tensor = _obs_to_tensor(next_obs, self.device)
            next_value = self.network.get_value(next_obs_tensor).reshape(1, -1).cpu()
            advantages = torch.zeros_like(self.rewards)
            last_gae_lam = 0.0
            for t in reversed(range(self.rollout_len)):
                next_non_terminal = 1.0 - (torch.as_tensor(next_done, dtype=torch.float32) if t == self.rollout_len - 1 else self.dones[t + 1])
                next_values = next_value if t == self.rollout_len - 1 else self.values[t + 1]
                delta = self.rewards[t] + self.config["gamma"] * next_values * next_non_terminal - self.values[t]
                advantages[t] = last_gae_lam = delta + self.config["gamma"] * self.config["gae_lambda"] * next_non_terminal * last_gae_lam
            returns = advantages + self.values

        b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
        b_log_probs = self.log_probs.reshape(-1)
        b_actions = self.actions.reshape((-1,) + self.env.action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        
        b_inds = np.arange(self.batch_size)
        for _ in range(self.config["num_update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                mb_obs = b_obs[mb_inds].to(self.device).float()
                mb_actions = b_actions[mb_inds].to(self.device).long()
                mb_log_probs = b_log_probs[mb_inds].to(self.device)
                mb_advantages = b_advantages[mb_inds].to(self.device)
                mb_returns = b_returns[mb_inds].to(self.device)

                _, new_log_prob, entropy, new_value = self.network.get_action_and_value(mb_obs, mb_actions)
                log_ratio = new_log_prob - mb_log_probs
                ratio = torch.exp(log_ratio)
                
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                if self.config["normalize_advantages"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                pg_loss = -torch.min(mb_advantages * ratio, mb_advantages * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])).mean()
                v_loss = 0.5 * ((new_value.view(-1) - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()
                loss = pg_loss - self.config["entropy_coef"] * entropy_loss + v_loss * self.config["value_loss_coef"]

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config["gradient_clip_norm"])
                self.optimizer.step()

        return {"loss": loss.item(), "policy_loss": pg_loss.item(), "value_loss": v_loss.item(), "entropy": entropy_loss.item(), "approx_kl": approx_kl.item()}

    def save(self, path: str):
        torch.save(self.network.state_dict(), path)

    def load(self, path: str):
        self.network.load_state_dict(torch.load(path))