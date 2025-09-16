# A Comparative Study of Value-Based and Policy-Based Deep Reinforcement Learning for Playing Atari Pong

This repository contains the implementation of the experiments conducted for the written assignment of the **DLMAIRIL01** course at IU International University of Applied Sciences.
The assignment is titled:

> *A Comparative Study of Value-Based and Policy-Based Deep Reinforcement Learning for Playing Atari Pong*

The codebase implements and compares **value-based** (DQN, with enhancements) and **policy-based** (PPO) reinforcement learning agents on the Atari Pong environment. It includes training pipelines, evaluation scripts, logging, and reproducibility support.

---

## Project Structure

The repository is organized into the following core components:

* **`agents/`**

  * `dqn_agent.py`: Implementation of a DQN agent with Double DQN and Dueling architecture.
  * `ppo_agent.py`: Implementation of a PPO agent with actor-critic architecture.

* **`models/`**

  * `dqn_model.py`: CNN + (dueling) head used by the DQN agent.
  * `ppo_model.py`: Shared CNN backbone with actor and critic heads for PPO.

* **`configs/`**

  * `dqn_vanilla.yaml`: Baseline DQN (ablation study).
  * `dqn_enhanced.yaml`: Enhanced DQN (Double + Dueling).
  * `ppo.yaml`: PPO configuration.

* **`utils/`**

  * `wrappers.py`: Standard Atari preprocessing (grayscale, frame-stacking, reward clipping).
  * `replay_buffer.py`: FIFO replay buffer for DQN.
  * `experiment_manager.py`: Manages checkpoints, metrics, and summaries for each run.
  * `logger.py`: Centralized logging of metrics across runs.

* **`scripts/`**

  * `train.py`: Main training loop with evaluation, checkpointing, and robustness tests.
  * `evaluate.py`: Run a trained model interactively to watch it play Pong.

* **`Makefile`**

  * Simplifies setup, training, evaluation, and monitoring.

---

## Setup

Ensure you have **Python 3.10+** installed.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/thomasreinecke/DLMAIRIL01-PlayingPong.git
   cd DLMAIRIL01-PlayingPong
   ```

2. **Setup virtual environment and install dependencies:**

   ```bash
   make install
   ```

   This will create a `.venv/` environment and install all required packages from `requirements.txt`.

---

## Running Pre-Trained Agents

Pre-trained models are stored in `output/<agent>_<seed>/snapshots/`.

To **watch an agent play Pong**, use the `watch` target:

```bash
make watch model=ppo_42
```

This will:

* Load the corresponding checkpoint (e.g., `output/ppo_42/snapshots/checkpoint.pt` or `model_final.pt`).
* Launch the Pong environment with rendering enabled.
* Run a number of episodes to watch the agent play.

---

## Training Agents

Training is fully managed via the **Makefile**.
Each run logs metrics to `output/<agent>_<seed>/results/` and checkpoints models in `output/<agent>_<seed>/snapshots/`.

Available training targets:

* **Train Vanilla DQN (ablation baseline):**

  ```bash
  make train-dqn-vanilla
  ```

* **Train Enhanced DQN (Double + Dueling):**

  ```bash
  make train-dqn-enhanced
  ```

* **Train PPO:**

  ```bash
  make train-ppo
  ```

* **Train all agents from scratch:**

  ```bash
  make train-all
  ```

Training parameters (frames, seeds, environment) are controlled at the top of the `Makefile`.

---

## Monitoring

* **TensorBoard:**

  ```bash
  make tensorboard
  ```

  Opens TensorBoard on `http://localhost:6006` to visualize learning curves, losses, and other metrics.

* **Jupyter Notebook:**

  ```bash
  make notebook
  ```

  Opens `notebooks/analysis.ipynb` for in-depth analysis.

---

## Most Relevant Makefile Targets

* `install`: Create virtual environment and install dependencies.
* `train-dqn-vanilla`: Train vanilla DQN across three seeds.
* `train-dqn-enhanced`: Train enhanced DQN (Double + Dueling) across three seeds.
* `train-ppo`: Train PPO agent across three seeds.
* `train-all`: Run all experiments from scratch.
* `watch`: Run a given pre-trained agent and render Pong.
* `tensorboard`: Launch TensorBoard.
* `notebook`: Open Jupyter notebook for analysis.
* `clean`: Remove all build artifacts and output data.
* `rebuild`: Clean everything and reinstall.

---

## Notes

* **Environment:** The project uses `ALE/Pong-v5` from Gymnasium with sticky actions (`p=0.25`) for evaluation consistency.
* **Reproducibility:** Seeds are fixed for NumPy, PyTorch, and environments, though some minor nondeterminism may remain on Apple MPS.
* **Logging:** Metrics are written to CSV and TensorBoard for later analysis. Final results are summarized in `final.json` per run.
