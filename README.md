# A Comparative Study of RL Agents for Atari Pong

This repository contains the source code and experimental setup for the paper "A Comparative Study of Value-Based and Policy-Based Deep Reinforcement Learning for Playing Atari Pong".

## Setup

1.  **Create Environment and Install Dependencies:**
    ```bash
    make setup
    ```
2.  **Activate Virtual Environment:**
    ```bash
    source .venv/bin/activate
    ```

## Training

To run all experiments for the paper (3 seeds for each of the 3 agents), use:
```bash
make train-all
```
Individual training runs can be launched for specific agents:

# Train Enhanced DQN for all seeds

```
make train-dqn-enhanced
```

# Train Vanilla DQN (ablation) for all seeds

```
make train-dqn-vanilla
```

# Train PPO for all seeds

```
make train-ppo
```

# Monitoring and Analysis
To monitor training progress in real-time:

```
make tensorboard
```

To analyze the final results and generate plots for the paper:

```
make notebook
```