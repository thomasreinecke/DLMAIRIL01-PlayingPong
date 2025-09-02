# scripts/train.py
import argparse
import os
import random
import time
from datetime import datetime
from typing import Dict
import sys
import platform
from collections import deque

# -----------------------------------------------------------
# Make sure PyTorch will gracefully fall back if an op
# isn't supported on MPS (Apple GPU).
# NOTE: Set this BEFORE importing torch.
# -----------------------------------------------------------
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Add project root to sys.path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import yaml
from gymnasium.spaces import Discrete

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.logger import Logger
from utils.wrappers import make_env


def select_device():
    """
    Prefer Apple MPS on macOS, then CUDA, then CPU.
    Returns (torch.device, human_readable_name).
    """
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps"), "mps (Apple Silicon)"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return torch.device("cuda"), f"cuda ({name})"
    return torch.device("cpu"), "cpu"


def log_device_info(logger: Logger, device: torch.device, device_name: str):
    """
    Log device info both as text (easy to read in TB) and as a few scalars.
    """
    lines = []
    lines.append(f"python: {platform.python_version()}")
    lines.append(f"pytorch: {torch.__version__}")
    lines.append(f"device: {device_name}")
    lines.append(f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK','<unset>')}")

    if device.type == "cuda":
        try:
            lines.append(f"cuda_capability: {torch.cuda.get_device_capability(0)}")
            lines.append(f"cuda_device_count: {torch.cuda.device_count()}")
            lines.append(f"cuda_current_device: {torch.cuda.current_device()}")
            total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            lines.append(f"cuda_total_mem_gb: {total:.2f}")
        except Exception as e:
            lines.append(f"cuda_info_error: {e!r}")

    if device.type == "mps" and hasattr(torch, "mps"):
        try:
            if hasattr(torch.mps, "current_allocated_memory"):
                mem = torch.mps.current_allocated_memory() / (1024 ** 2)
                lines.append(f"mps_current_allocated_mb: {mem:.2f}")
            if hasattr(torch.mps, "driver_allocated_memory"):
                dmem = torch.mps.driver_allocated_memory() / (1024 ** 2)
                lines.append(f"mps_driver_allocated_mb: {dmem:.2f}")
        except Exception as e:
            lines.append(f"mps_info_error: {e!r}")

    logger.writer.add_text("system/device_info", "```\n" + "\n".join(lines) + "\n```", global_step=0)
    logger.log_scalar("system/device/is_mps", 1.0 if device.type == "mps" else 0.0, step=0)
    logger.log_scalar("system/device/is_cuda", 1.0 if device.type == "cuda" else 0.0, step=0)
    logger.log_scalar("system/device/is_cpu", 1.0 if device.type == "cpu" else 0.0, step=0)


def get_agent(env, config, logger, device):
    """Factory function to create an agent based on config."""
    if config["algo"] == "dqn":
        return DQNAgent(env, config, logger, device)
    elif config["algo"] == "ppo":
        return PPOAgent(env, config, logger, device)
    else:
        raise ValueError(f"Unknown algorithm: {config['algo']}")


def evaluate_agent(env, agent, num_episodes: int):
    """Evaluate the agent's performance."""
    total_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
        total_rewards.append(episode_reward)
    return np.mean(total_rewards), np.std(total_rewards)


def main(args):
    # --- 1. Load configuration ---
    with open(args.config, "r") as f:
        config: Dict = yaml.safe_load(f)

    run_name = f"{config['agent_name']}__{args.seed}__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = os.path.join("runs", run_name)
    data_dir = os.path.join("data", run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    logger = Logger(log_dir)
    logger.log_hyperparams(config)

    # --- 2. Device & seeding ---
    device, device_name = select_device()
    print(f"Using device: {device} ({device_name})")
    log_device_info(logger, device, device_name)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # --- 3. Environments ---
    env = make_env(args.env_id, args.seed)
    eval_env = make_env(args.env_id, args.seed + 100)

    if not isinstance(env.action_space, Discrete):
        raise ValueError("This project expects a discrete action space (Atari).")

    agent = get_agent(env, config, logger, device)

    # --- 4. Training loop ---
    start_time = time.time()
    obs, info = env.reset(seed=args.seed)
    next_eval_frame = config["eval_freq_frames"]

    # Heartbeat state
    heartbeat_secs = max(1, int(args.heartbeat_secs))
    last_hb_time = time.time()
    last_hb_frame = 0

    # Episode rolling stats
    ep_returns = deque(maxlen=20)
    ep_lengths = deque(maxlen=20)

    # Last known train metrics
    last_train_loss = None
    last_policy_loss = None
    last_value_loss = None
    last_entropy = None

    for frame_idx in range(1, args.frames + 1):
        is_ppo = config["algo"] == "ppo"
        is_dqn = config["algo"] == "dqn"

        # Agent acts and collects experience
        if is_ppo:
            action = agent.collect_rollout(obs, np.array([False]))
        else:
            action = agent.act(obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store experience
        if is_dqn:
            agent.add_experience(obs, action, reward, next_obs, done)
        else:
            agent.set_reward([reward])

        obs = next_obs
        if done:
            if "episode" in info:
                ep_r = info["episode"]["r"]
                ep_l = info["episode"]["l"]
                ep_returns.append(float(ep_r[0]) if hasattr(ep_r, "__len__") else float(ep_r))
                ep_lengths.append(int(ep_l[0]) if hasattr(ep_l, "__len__") else int(ep_l))
            obs, info = env.reset()

        # Training step
        if is_dqn and frame_idx > config["learning_starts"]:
            train_info = agent.train()
            if train_info and "loss" in train_info:
                last_train_loss = float(train_info["loss"])
                logger.log_scalar("charts/loss", last_train_loss, frame_idx)

        if is_ppo:
            agent.finish_step()
            if agent.should_train():
                train_info = agent.train(next_obs, np.array([done]))
                # Cache last train metrics for heartbeat display
                last_train_loss = float(train_info.get("loss", np.nan))
                last_policy_loss = float(train_info.get("policy_loss", np.nan))
                last_value_loss = float(train_info.get("value_loss", np.nan))
                last_entropy = float(train_info.get("entropy", np.nan))
                for key, val in train_info.items():
                    logger.log_scalar(f"charts/{key}", float(val), frame_idx)

        # Evaluation cadence (frame-based)
        if frame_idx >= next_eval_frame:
            print(f"--- Evaluating at frame {frame_idx}/{args.frames} ---")
            mean_reward, std_reward = evaluate_agent(eval_env, agent, config["eval_episodes"])
            print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

            logger.log_scalar("eval/mean_return", mean_reward, frame_idx)
            logger.log_scalar("eval/std_return", std_reward, frame_idx)

            fps = int(frame_idx / (time.time() - start_time))
            logger.log_scalar("charts/fps", fps, frame_idx)

            logger.write_log()  # CSV flush
            next_eval_frame += config["eval_freq_frames"]

        # ------------------------------
        # Time-based heartbeat logging
        # ------------------------------
        now = time.time()
        if now - last_hb_time >= heartbeat_secs:
            elapsed = now - start_time
            fps_overall = frame_idx / max(1e-6, elapsed)
            fps_window = (frame_idx - last_hb_frame) / max(1e-6, (now - last_hb_time))

            logger.log_scalar("heartbeat/frame", frame_idx, frame_idx)
            logger.log_scalar("heartbeat/elapsed_seconds", elapsed, frame_idx)
            logger.log_scalar("heartbeat/fps_overall", fps_overall, frame_idx)
            logger.log_scalar("heartbeat/fps_window", fps_window, frame_idx)

            # Episode rolling means (if any episodes finished)
            if ep_returns:
                logger.log_scalar("heartbeat/episode_return_ma20", float(np.mean(ep_returns)), frame_idx)
            if ep_lengths:
                logger.log_scalar("heartbeat/episode_length_ma20", float(np.mean(ep_lengths)), frame_idx)

            # Last train losses (if known)
            if last_train_loss is not None:
                logger.log_scalar("heartbeat/train_loss_last", last_train_loss, frame_idx)
            if is_ppo:
                if last_policy_loss is not None:
                    logger.log_scalar("heartbeat/ppo/policy_loss_last", last_policy_loss, frame_idx)
                if last_value_loss is not None:
                    logger.log_scalar("heartbeat/ppo/value_loss_last", last_value_loss, frame_idx)
                if last_entropy is not None:
                    logger.log_scalar("heartbeat/ppo/entropy_last", last_entropy, frame_idx)

            # Algo-specific quick stats
            if is_dqn:
                logger.log_scalar("heartbeat/dqn/epsilon", float(agent._get_current_epsilon()), frame_idx)
                try:
                    # Replay buffer length
                    logger.log_scalar("heartbeat/dqn/replay_buffer_size", float(len(agent.replay_buffer)), frame_idx)
                except Exception:
                    pass

            # Device memory stats (best-effort)
            if device.type == "cuda":
                try:
                    used = torch.cuda.memory_allocated() / (1024 ** 2)
                    logger.log_scalar("system/memory/cuda_allocated_mb", used, frame_idx)
                except Exception:
                    pass
            elif device.type == "mps" and hasattr(torch, "mps"):
                try:
                    if hasattr(torch.mps, "current_allocated_memory"):
                        used = torch.mps.current_allocated_memory() / (1024 ** 2)
                        logger.log_scalar("system/memory/mps_allocated_mb", used, frame_idx)
                except Exception:
                    pass

            # Flush both CSV and TensorBoard to disk
            logger.write_log()
            logger.flush()

            # Reset heartbeat window
            last_hb_time = now
            last_hb_frame = frame_idx

    # --- 5. Final save and cleanup ---
    model_path = os.path.join(data_dir, f"{config['agent_name']}_final.pth")
    agent.save(model_path)
    print(f"Model saved to {model_path}")
    env.close()
    eval_env.close()
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the experiment")
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5", help="Gymnasium environment ID")
    parser.add_argument("--frames", type=int, default=5_000_000, help="Total number of frames to train for")
    parser.add_argument("--heartbeat-secs", type=int, default=10, help="Seconds between heartbeat logs")
    args = parser.parse_args()
    main(args)
