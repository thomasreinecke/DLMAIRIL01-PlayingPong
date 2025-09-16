# scripts/evaluate.py
import argparse
import time
import os
import sys

# Add project root to sys.path so imports work when running as a script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import yaml

from scripts.train import get_agent
from utils.wrappers import make_env


def main(args):
    # --- 1. Load configuration and model ---
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Choose device: CUDA if available, else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create environment with rendering enabled
    env = make_env(args.env_id, args.seed, render_mode="human")

    # Instantiate agent from config + load pretrained model weights
    agent = get_agent(env, config, device)
    agent.load(args.model_path)

    print(f"Loaded model from {args.model_path}")
    print("Starting evaluation...")

    # --- 2. Run evaluation loop ---
    for episode in range(args.episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()  # show environment window
            # Use deterministic policy (no exploration noise) during evaluation
            action = agent.act(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            time.sleep(1 / 60)  # Limit FPS to ~60Hz for smoother rendering

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required paths
    parser.add_argument("--config", type=str, required=True, help="Path to the agent's config file")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.pth) file")
    # Environment options
    parser.add_argument("--env-id", type=str, default="ALE/PongNoFrameskip-v5", help="Gymnasium environment ID")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for the environment")
    # Evaluation options
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to run")
    args = parser.parse_args()
    main(args)
