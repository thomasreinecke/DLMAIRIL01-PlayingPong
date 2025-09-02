# utils/experiment_manager.py
import os
import json
import csv
import time
from datetime import datetime
from typing import Dict, Any, List

import torch
from torch.utils.tensorboard import SummaryWriter


class ExperimentManager:
    """Manages all artifacts for a single experiment run in a structured directory."""

    def __init__(self, agent_name: str, seed: int):
        self.run_name = f"{agent_name}_{seed}"
        self.base_dir = os.path.join("output", self.run_name)
        self.agent_name = agent_name

        # Create the full directory structure
        self.results_dir = os.path.join(self.base_dir, "results")
        self.snapshots_dir = os.path.join(self.base_dir, "snapshots")
        self.tensorboard_dir = os.path.join(self.base_dir, "tensorboard")
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)
        os.makedirs(self.tensorboard_dir, exist_ok=True)

        # Define all artifact paths
        self.checkpoint_path = os.path.join(self.snapshots_dir, "checkpoint.pt")
        self.final_model_path = os.path.join(self.snapshots_dir, "model_final.pt")
        self.status_path = os.path.join(self.results_dir, "status.json")
        self.results_csv_path = os.path.join(self.results_dir, "metrics.csv")
        self.final_summary_path = os.path.join(self.results_dir, "final.json")

        # In-memory state
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        self._last_status_update_time = 0

        # --- THIS IS THE FIX ---
        # Define agent-specific headers for the CSV file.
        self._define_csv_headers()

    def _define_csv_headers(self):
        """Selects the correct CSV headers based on the agent type."""
        core_headers = [
            'environment_frame', 'wall_clock_time_seconds', 'fps',
            'eval_mean_return', 'eval_std_return'
        ]
        if 'dqn' in self.agent_name:
            self.csv_headers = core_headers + ['epsilon', 'q_learning_loss', 'mean_q_value']
        elif 'ppo' in self.agent_name:
            self.csv_headers = core_headers + ['policy_loss', 'value_loss', 'entropy', 'approx_kl']
        else:
            self.csv_headers = core_headers

        # Create the CSV with the correct headers if it doesn't exist.
        if not os.path.exists(self.results_csv_path):
            with open(self.results_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.csv_headers)
                writer.writeheader()


    def update_status(self, phase: str, frame_idx: int, total_frames: int):
        """Writes the current status to status.json, throttled for high-frequency phases."""
        now = time.time()
        if phase == "collecting_experience" and now - self._last_status_update_time < 1.0:
            return
        
        percent_complete = (frame_idx / total_frames) * 100 if total_frames > 0 else 0
        status_data = {
            "phase": phase,
            "last_update": datetime.now().isoformat(),
            "current_frame": frame_idx,
            "percent_complete": f"{percent_complete:.2f}%",
        }
        with open(self.status_path, "w") as f:
            json.dump(status_data, f, indent=4)
        self._last_status_update_time = now

    def save_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer, frame_idx: int):
        """Saves a resumable checkpoint."""
        checkpoint = {
            "frame_idx": frame_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"  > Checkpoint saved to {self.checkpoint_path}")

    def save_final_model(self, model: torch.nn.Module):
        """Saves just the final model weights."""
        torch.save(model.state_dict(), self.final_model_path)
        print(f"  > Final model saved to {self.final_model_path}")

    def load_checkpoint(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
        """Loads state from a checkpoint. Returns the frame to start from."""
        if not os.path.exists(self.checkpoint_path):
            print("  > No checkpoint found, starting from scratch.")
            return 0

        print(f"  > Loading checkpoint from {self.checkpoint_path}")
        checkpoint = torch.load(self.checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_frame = checkpoint["frame_idx"] + 1
        print(f"  > Resuming training from frame {start_frame}")
        return start_frame

    def log_metrics(self, frame_idx: int, metrics: Dict[str, Any]):
        """Logs metrics to the run-specific results.csv and TensorBoard."""
        with open(self.results_csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_headers)
            row = {k: metrics.get(k) for k in self.csv_headers}
            writer.writerow(row)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"metric/{key}", value, frame_idx)

    def save_final_summary(self, summary_data: Dict[str, Any]):
        """Saves the final summary dictionary to final.json."""
        with open(self.final_summary_path, "w") as f:
            json.dump(summary_data, f, indent=4)
        print(f"  > Final summary saved to {self.final_summary_path}")

    def close(self):
        self.writer.flush()
        self.writer.close()