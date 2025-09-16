# utils/wrappers.py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformReward, RecordEpisodeStatistics
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack


def make_env(env_id: str, seed: int, render_mode: str = None, sticky_actions_prob: float = 0.25):
    """
    Create and wrap the Atari environment with a standard stack of wrappers.

    - Ensures consistent seeding for reproducibility.
    - Applies common Atari preprocessing (grayscale, resize to 84x84, no frame skipping).
    - Clips rewards to [-1, 1] for stable learning.
    - Stacks 4 frames to provide temporal context to the agent.
    """
    # Create base Atari env.
    # NOTE: v5 environments (e.g., ALE/Pong-v5) default to sticky actions with p=0.25.
    #       To override, pass `repeat_action_probability`.
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=sticky_actions_prob)

    # Seed action and observation spaces for deterministic behavior under the given seed
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Collect episode returns and lengths automatically (added to `info["episode"]`)
    env = RecordEpisodeStatistics(env)

    # Apply standard Atari preprocessing:
    # - Grayscale observations
    # - Resize frames to 84x84
    # - Disable frame skip (since ALE/Pong-v5 already skips internally)
    # - Disable observation scaling (we handle normalization in the agent)
    # - No early termination on life loss
    env = AtariPreprocessing(
        env,
        noop_max=30,                # up to 30 no-ops at start to randomize initial states
        frame_skip=1,               # avoid double-skipping (env already applies skip)
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    # Clip rewards to [-1, 1] to stabilize learning (classic DQN/PPO setup for Atari)
    env = TransformReward(env, lambda r: float(np.clip(r, -1.0, 1.0)))

    # Stack 4 consecutive frames into a single observation tensor
    # Provides temporal information about ball/paddle movement
    env = FrameStack(env, 4)

    return env
