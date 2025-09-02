# utils/logger.py
import csv
import os
from typing import Any, Dict

from torch.utils.tensorboard import SummaryWriter
import yaml


class Logger:
    """
    Lightweight logger that writes:
      - TensorBoard scalars (no add_hparams to avoid NumPy 2.0 issues).
      - A human-readable hparams YAML (hparams.yaml) and a TB text entry.
      - Optional CSV dump of logged scalars via write_log().
    """

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(self.log_dir)
        self._buffer = []  # list[dict]: {"step": int, "tag": str, "value": float}
        self._csv_path = os.path.join(self.log_dir, "metrics.csv")

    # ----------------------------
    # Hyperparameters
    # ----------------------------
    def log_hyperparams(self, hparams: Dict[str, Any]) -> None:
        """
        Store hyperparameters in:
          - log_dir/hparams.yaml
          - TensorBoard as a text block ("hparams")
        Avoids torch.utils.tensorboard.add_hparams to prevent NumPy 2.0 breakage.
        """
        # 1) Dump YAML file
        hparams_yaml_path = os.path.join(self.log_dir, "hparams.yaml")
        with open(hparams_yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(_to_serializable(hparams), f, sort_keys=True, allow_unicode=True)

        # 2) Also put them into TB as readable text
        with open(hparams_yaml_path, "r", encoding="utf-8") as f:
            text = f.read()
        self.writer.add_text("hparams", f"```yaml\n{text}\n```", global_step=0)

        # 3) Log numeric hparams individually as scalars at step 0
        flat = _flatten_dict(hparams)
        for k, v in flat.items():
            if isinstance(v, (int, float, bool)):
                self.writer.add_scalar(f"hparams/{k}", float(v) if isinstance(v, bool) else v, global_step=0)

        self.writer.flush()

    # ----------------------------
    # Scalars
    # ----------------------------
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)
        self._buffer.append({"step": step, "tag": tag, "value": float(value)})

    # ----------------------------
    # CSV flush
    # ----------------------------
    def write_log(self) -> None:
        """Append buffered scalar logs to CSV (log_dir/metrics.csv)."""
        if not self._buffer:
            return

        # Ensure deterministic ordering by (step, tag)
        records = sorted(self._buffer, key=lambda r: (r["step"], r["tag"]))
        self._buffer = []

        header = ["step", "tag", "value"]
        file_exists = os.path.exists(self._csv_path)
        with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            for r in records:
                writer.writerow(r)

    # ----------------------------
    # Lifecycle
    # ----------------------------
    def flush(self) -> None:
        """Flush TB writer (useful for periodic heartbeats)."""
        self.writer.flush()

    def close(self) -> None:
        # Flush any pending CSV writes
        self.write_log()
        self.writer.flush()
        self.writer.close()


# ----------------------------
# Helpers
# ----------------------------
def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = "/") -> Dict[str, Any]:
    """Flatten nested dicts using 'parent/child' keys."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _to_serializable(obj: Any) -> Any:
    """
    Convert objects that may not be YAML-serializable (e.g., numpy types)
    to plain Python types.
    """
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(x) for x in obj]
    # Convert basic numpy / tensor-like scalars if they sneak in
    try:
        if hasattr(obj, "item"):
            return obj.item()
    except Exception:
        pass
    return obj
