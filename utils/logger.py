# utils/logger.py
import csv
import os
import fcntl  # For file locking on Unix-like systems
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter

class CentralLogger:
    """
    Manages a single, unified CSV file for all experiment results,
    and also writes to a run-specific TensorBoard log directory.
    """

    def __init__(self, csv_filepath="results.csv", tensorboard_log_dir: str = None):
        self.csv_filepath = csv_filepath
        self.lock_path = csv_filepath + ".lock"
        
        # Setup TensorBoard writer if a directory is provided
        self.writer = SummaryWriter(log_dir=tensorboard_log_dir) if tensorboard_log_dir else None
        
        self.headers = [
            # Core Columns
            'agent_name', 'seed', 'environment_frame', 'wall_clock_time_seconds', 'fps',
            'eval_mean_return', 'eval_std_return',
            # DQN-Specific
            'epsilon', 'q_learning_loss', 'mean_q_value',
            # PPO-Specific
            'policy_loss', 'value_loss', 'entropy', 'approx_kl',
            # Robustness
            'robustness_eval_p0.0_return', 'robustness_eval_p0.5_return',
        ]

        # Create the CSV file with header if it doesn't exist
        if not os.path.exists(self.csv_filepath):
            with open(self.csv_filepath, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writeheader()

    def log_metrics(self, data: Dict[str, Any], frame_idx: int):
        """
        Logs a dictionary of metrics to both the central CSV and TensorBoard.
        """
        # --- 1. Log to Central CSV ---
        # Acquire a lock to prevent concurrent writes
        with open(self.lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            # Create a full row, filling missing keys with None
            row = {header: data.get(header) for header in self.headers}
            
            with open(self.csv_filepath, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(row)

            # Release the lock
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            
        # --- 2. Log to TensorBoard ---
        if self.writer:
            for key, value in data.items():
                # Only log numeric values to TensorBoard scalars
                if isinstance(value, (int, float)):
                    self.writer.add_scalar(f"metric/{key}", value, frame_idx)
    
    def close(self):
        """Closes the TensorBoard writer."""
        if self.writer:
            self.writer.flush()
            self.writer.close()