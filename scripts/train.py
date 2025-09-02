# scripts/train.py
import argparse
import os
import random
import time
from typing import Dict
import sys

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import torch
import yaml

from agents.dqn_agent import DQNAgent
from agents.ppo_agent import PPOAgent
from utils.experiment_manager import ExperimentManager
from utils.wrappers import make_env

def select_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "mps (Apple Silicon)"
    if torch.cuda.is_available():
        return torch.device("cuda"), f"cuda ({torch.cuda.get_device_name(0)})"
    return torch.device("cpu"), "cpu"

def get_agent(env, config, device):
    if config["algo"] == "dqn":
        return DQNAgent(env, config, device)
    elif config["algo"] == "ppo":
        return PPOAgent(env, config, device)
    raise ValueError(f"Unknown algorithm: {config['algo']}")

def evaluate_agent(env, agent, num_episodes: int):
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

def generate_final_summary(manager: ExperimentManager, config: Dict, args: argparse.Namespace, total_time: float, robustness_results: Dict) -> Dict:
    """Calculates all summary statistics and builds the final.json object."""
    print("\n--- Generating Final Summary ---")
    summary = {}
    
    try:
        df = pd.read_csv(manager.results_csv_path)
        eval_df = df.dropna(subset=['eval_mean_return']).copy()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        print("Warning: metrics.csv is empty or not found. Summary will be incomplete.")
        eval_df = pd.DataFrame()

    summary['agent_name'] = config['agent_name']
    summary['seed'] = args.seed
    
    summary['training_parameters'] = {
        'total_frames': args.frames,
        'final_epsilon': agent._get_current_epsilon() if config['algo'] == 'dqn' else None,
        'learning_rate': config['learning_rate'],
        'environment_id': args.env_id
    }

    perf_summary = {}
    if not eval_df.empty:
        last_5_evals = eval_df.nlargest(5, 'environment_frame')
        perf_summary['final_score_mean'] = last_5_evals['eval_mean_return'].mean()
        perf_summary['final_score_std'] = last_5_evals['eval_mean_return'].std()

        THRESHOLD = 15.0
        threshold_reached_df = eval_df[eval_df['eval_mean_return'] >= THRESHOLD]
        perf_summary['frames_to_threshold'] = int(threshold_reached_df['environment_frame'].min()) if not threshold_reached_df.empty else None
            
        eval_df.loc[:, 'eval_mean_return_clipped'] = eval_df['eval_mean_return'].clip(lower=-21.0)
        perf_summary['auc_at_5M_frames'] = np.trapz(
            y=eval_df['eval_mean_return_clipped'], x=eval_df['environment_frame']
        )
    summary['performance_summary'] = perf_summary

    comp_summary = {
        'total_wall_clock_time_seconds': total_time,
        'mean_fps': args.frames / total_time if total_time > 0 else 0
    }
    if not eval_df.empty and perf_summary.get('frames_to_threshold') is not None:
        frame = perf_summary['frames_to_threshold']
        time_to_thresh_row = eval_df[eval_df['environment_frame'] >= frame].iloc[0]
        comp_summary['time_to_threshold_seconds'] = time_to_thresh_row['wall_clock_time_seconds']
    else:
        comp_summary['time_to_threshold_seconds'] = None
    summary['computational_summary'] = comp_summary

    summary['robustness_summary'] = {
        'eval_return_sticky_p0_0': robustness_results.get(0.0, {}).get('mean'),
        'eval_return_sticky_p0_25': perf_summary.get('final_score_mean'),
        'eval_return_sticky_p0_5': robustness_results.get(0.5, {}).get('mean')
    }
    
    summary['final_model_path'] = manager.final_model_path
    
    return summary

def main(args):
    with open(args.config, "r") as f:
        config: Dict = yaml.safe_load(f)

    manager = ExperimentManager(config['agent_name'], args.seed)
    manager.update_status("initializing", 0, args.frames)
    
    device, device_name = select_device()
    print(f"Using device: {device} ({device_name})")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args.env_id, args.seed, sticky_actions_prob=0.25)
    global agent
    agent = get_agent(env, config, device)
    
    start_frame = manager.load_checkpoint(agent.network, agent.optimizer)
    
    if start_frame >= args.frames:
        print("Training already completed. Exiting.")
        manager.update_status("completed", args.frames, args.frames)
        return

    print(f"\n--- Starting/Resuming training for {manager.run_name} ---")
    
    overall_start_time = time.time()
    obs, info = env.reset(seed=args.seed)
    
    if start_frame > 0 and os.path.exists(manager.results_csv_path):
        try:
            df = pd.read_csv(manager.results_csv_path)
            last_time = df['wall_clock_time_seconds'].iloc[-1]
            start_time = time.time() - last_time
        except Exception:
            start_time = time.time()
    else:
        start_time = time.time()

    next_eval_frame = (start_frame // config["eval_freq_frames"] + 1) * config["eval_freq_frames"]
    last_train_info = {}
    agent.frame_idx = start_frame

    for frame_idx in range(start_frame, args.frames + 1):
        manager.update_status("collecting_experience", frame_idx, args.frames)
        agent.frame_idx = frame_idx
        is_ppo = config["algo"] == "ppo"
        is_dqn = config["algo"] == "dqn"

        action = agent.act(obs) if is_dqn else agent.collect_rollout(obs, np.array([False]))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if is_dqn:
            agent.add_experience(obs, action, reward, next_obs, done)
        else:
            agent.set_reward([reward])

        obs = next_obs
        if done:
            obs, info = env.reset()
        
        update_occurred = False
        if is_dqn and frame_idx > config["learning_starts"] and frame_idx % 4 == 0:
            manager.update_status("updating_policy", frame_idx, args.frames)
            last_train_info.update(agent.train())
            update_occurred = True
        if is_ppo and agent.should_train():
            agent.finish_step()
            manager.update_status("updating_policy", frame_idx, args.frames)
            last_train_info.update(agent.train(next_obs, np.array([done])))
            update_occurred = True
        elif is_ppo:
            agent.finish_step()
        
        if update_occurred:
             manager.update_status("collecting_experience", frame_idx, args.frames)

        if frame_idx >= next_eval_frame and frame_idx > 0:
            manager.update_status("evaluating", frame_idx, args.frames)
            eval_env = make_env(args.env_id, args.seed + 100, sticky_actions_prob=0.25)
            mean_ret, std_ret = evaluate_agent(eval_env, agent, config["eval_episodes"])
            eval_env.close()

            log_data = {
                "environment_frame": frame_idx,
                "wall_clock_time_seconds": time.time() - start_time,
                "fps": frame_idx / (time.time() - start_time + 1e-9),
                "eval_mean_return": mean_ret,
                "eval_std_return": std_ret,
            }
            log_data.update(last_train_info)
            if is_dqn:
                log_data["epsilon"] = agent._get_current_epsilon()
            
            manager.log_metrics(frame_idx, log_data)
            print(f"  > Eval @ {frame_idx/1e6:.2f}M frames: Mean Return = {mean_ret:.2f}")

            manager.update_status("saving_checkpoint", frame_idx, args.frames)
            manager.save_checkpoint(agent.network, agent.optimizer, frame_idx)
            next_eval_frame += config["eval_freq_frames"]

    total_training_time = time.time() - overall_start_time
    print(f"\n--- Training complete for {manager.run_name}. Running robustness tests... ---")
    
    robustness_results = {}
    for prob in [0.0, 0.5]:
        print(f"  > Evaluating with sticky_actions_prob = {prob}...")
        robust_env = make_env(args.env_id, args.seed + 200, sticky_actions_prob=prob)
        mean_ret, std_ret = evaluate_agent(robust_env, agent, num_episodes=30)
        robustness_results[prob] = {"mean": mean_ret, "std": std_ret}
        print(f"    - Result: {mean_ret:.2f} +/- {std_ret:.2f}")
        robust_env.close()
    
    final_summary = generate_final_summary(manager, config, args, total_training_time, robustness_results)
    manager.save_final_summary(final_summary)
    
    manager.save_final_model(agent.network)
    manager.update_status("completed", args.frames, args.frames)
    print("--- Run finished. ---")
    
    env.close()
    manager.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RL agents with self-contained logging and checkpointing.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", type=str, default="ALE/Pong-v5")
    parser.add_argument("--frames", type=int, default=5_000_000)
    args = parser.parse_args()
    main(args)