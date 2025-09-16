# agents/ppo_agent.py
from typing import Dict
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from gymnasium import Env

from agents.base_agent import BaseAgent
from models.ppo_model import PPOActorCritic


def _obs_to_tensor(obs, device: torch.device) -> torch.Tensor:
    """Convert observation (LazyFrames/ndarray) into a normalized float32 tensor
    in [0,1] with batch dimension."""
    arr = np.asarray(obs, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).to(device)


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) Agent.
    - Uses an actor-critic model with policy + value head.
    - Implements rollout buffer for a single environment.
    """

    def __init__(self, env: Env, config: Dict, device: torch.device):
        self.env = env
        self.config = config
        self.device = device

        # Actor-Critic network and optimizer
        self.network = PPOActorCritic(env.observation_space, env.action_space).to(device)
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config["learning_rate"],
            eps=1e-5,
        )

        # Rollout parameters (single environment)
        self.num_envs = 1
        self.rollout_len = int(config["rollout_len"])
        self.batch_size = self.num_envs * self.rollout_len
        self.minibatch_size = self.batch_size // int(config["num_minibatches"])

        # Allocate rollout storage (CPU tensors)
        obs_shape = env.observation_space.shape
        self.obs = torch.zeros(
            (self.rollout_len, self.num_envs) + obs_shape, dtype=torch.uint8, device="cpu"
        )
        # Actions are discrete indices
        self.actions = torch.zeros((self.rollout_len, self.num_envs), dtype=torch.long, device="cpu")
        self.log_probs = torch.zeros((self.rollout_len, self.num_envs), dtype=torch.float32, device="cpu")
        self.rewards = torch.zeros((self.rollout_len, self.num_envs), dtype=torch.float32, device="cpu")
        self.dones = torch.zeros((self.rollout_len, self.num_envs), dtype=torch.float32, device="cpu")
        self.values = torch.zeros((self.rollout_len, self.num_envs), dtype=torch.float32, device="cpu")
        self.step = 0  # index within rollout buffer

    # ---------- interaction ----------
    def act(self, obs: np.ndarray, training: bool = True) -> int:
        """Select an action from the policy.
        - During training: sample stochastically.
        - During evaluation: pick argmax (greedy)."""
        obs_tensor = _obs_to_tensor(obs, self.device)
        with torch.no_grad():
            if training:
                action, _, _, _ = self.network.get_action_and_value(obs_tensor)
            else:
                logits = self.network.get_policy_logits(obs_tensor)
                action = logits.argmax(dim=-1)
        return int(action.squeeze().item())

    def collect_rollout(self, next_obs, next_done):
        """Collect one environment transition into rollout buffer.
        Stores obs, done flag, action, log-prob, and value estimate."""
        # Save observation + termination flag
        self.obs[self.step] = torch.as_tensor(np.asarray(next_obs), device="cpu")
        self.dones[self.step] = torch.as_tensor(next_done, dtype=torch.float32, device="cpu")

        # Query network for action, log-prob, and value
        with torch.no_grad():
            obs_tensor = (self.obs[self.step].to(self.device).float() / 255.0)
            action, log_prob, _, value = self.network.get_action_and_value(obs_tensor)
            self.values[self.step] = value.flatten().cpu()

        # Store action + log-prob
        self.actions[self.step] = action.cpu()
        self.log_probs[self.step] = log_prob.cpu()
        return int(action.item())

    def set_reward(self, reward: float):
        """Store environment reward for current step."""
        self.rewards[self.step] = torch.as_tensor(reward, dtype=torch.float32, device="cpu").view(-1)

    def finish_step(self):
        """Advance buffer step index (wraps after rollout_len)."""
        self.step = (self.step + 1) % self.rollout_len

    def should_train(self) -> bool:
        """Return True if a full rollout has been collected (ready to train)."""
        return self.step == 0

    # ---------- learning ----------
    def train(self, next_obs, next_done) -> Dict[str, float]:
        """Perform one PPO update using the collected rollout buffer."""
        T = self.rollout_len

        # --- compute advantages and returns (GAE-Lambda) ---
        with torch.no_grad():
            next_obs_tensor = _obs_to_tensor(next_obs, self.device)
            next_value = self.network.get_value(next_obs_tensor).reshape(1, -1).cpu()

            advantages = torch.zeros_like(self.rewards)
            last_gae_lam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    # bootstrap from next state
                    next_non_terminal = 1.0 - torch.as_tensor(next_done, dtype=torch.float32)
                    next_values = next_value
                else:
                    # NOTE: indexing must use self.dones[t]
                    next_non_terminal = 1.0 - self.dones[t]
                    next_values = self.values[t + 1]

                delta = self.rewards[t] + self.config["gamma"] * next_values * next_non_terminal - self.values[t]
                last_gae_lam = delta + self.config["gamma"] * self.config["gae_lambda"] * next_non_terminal * last_gae_lam
                advantages[t] = last_gae_lam

            returns = advantages + self.values

        # Flatten rollout tensors for training
        b_obs = self.obs.reshape((-1,) + self.env.observation_space.shape)
        b_actions = self.actions.reshape(-1)
        b_log_probs = self.log_probs.reshape(-1)
        b_values = self.values.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        # Normalize advantages if enabled (variance reduction)
        if self.config["normalize_advantages"]:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # --- stochastic gradient descent over minibatches ---
        b_inds = np.arange(self.batch_size)
        approx_kl = torch.tensor(0.0)
        entropy_loss = torch.tensor(0.0)
        pg_loss = torch.tensor(0.0)
        v_loss = torch.tensor(0.0)

        target_kl = float(self.config.get("target_kl", 0.0))  # optional early stopping

        for _ in range(self.config["num_update_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]

                # Move minibatch data to device + normalize obs
                mb_obs = (b_obs[mb_inds].to(self.device).float() / 255.0)
                mb_actions = b_actions[mb_inds].to(self.device)
                mb_old_logp = b_log_probs[mb_inds].to(self.device)
                mb_adv = b_advantages[mb_inds].to(self.device)
                mb_returns = b_returns[mb_inds].to(self.device)

                # Recompute action log-probs, entropy, and value
                _, new_logp, entropy, new_value = self.network.get_action_and_value(mb_obs, mb_actions)
                log_ratio = new_logp - mb_old_logp
                ratio = torch.exp(log_ratio)

                # KL diagnostic
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean()

                # PPO clipped objective
                unclipped = -mb_adv * ratio
                clipped = -mb_adv * torch.clamp(ratio, 1 - self.config["clip_coef"], 1 + self.config["clip_coef"])
                pg_loss = torch.max(unclipped, clipped).mean()

                # Value and entropy losses
                v_loss = 0.5 * ((new_value.view(-1) - mb_returns) ** 2).mean()
                entropy_loss = entropy.mean()

                # Combined PPO loss
                loss = pg_loss - self.config["entropy_coef"] * entropy_loss + self.config["value_loss_coef"] * v_loss

                # Gradient step
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config["gradient_clip_norm"])
                self.optimizer.step()

            # optional early stop if KL exceeds threshold
            if target_kl and approx_kl.item() > target_kl:
                break

        return {
            "loss": (pg_loss - self.config["entropy_coef"] * entropy_loss + self.config["value_loss_coef"] * v_loss).item(),
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": approx_kl.item(),
        }

    def save(self, path: str):
        """Save actor-critic network parameters to disk."""
        torch.save(self.network.state_dict(), path)


    def load(self, path: str):
        """Load parameters into online and target networks (synchronized)."""
        checkpoint = torch.load(path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.network.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.network.load_state_dict(checkpoint)