# scripts/evaluate.py
import argparse
import time
import os
import sys

# Add project root to sys.path to allow for package imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml

from scripts.train import get_agent
from utils.wrappers import make_env


def main(args):
    # --- 1. Load configuration and model ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(args.env_id, args.seed, render_mode="human")
    agent = get_agent(env, config, None, device)
    agent.load(args.model_path)

    print(f"Loaded model from {args.model_path}")
    print("Starting evaluation...")

    # --- 2. Run evaluation loop ---
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            env.render()
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            time.sleep(1 / 60)  # Slow down rendering
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the agent's config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth) file")
    parser.add_argument("--env-id", type=str, default="ALE/PongNoFrameskip-v5", help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()
    main(args)