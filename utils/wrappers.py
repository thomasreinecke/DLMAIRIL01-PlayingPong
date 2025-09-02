# utils/wrappers.py
import gymnasium as gym
import numpy as np
from gymnasium.wrappers import TransformReward, RecordEpisodeStatistics
from gymnasium.wrappers.atari_preprocessing import AtariPreprocessing
from gymnasium.wrappers.frame_stack import FrameStack


def make_env(env_id: str, seed: int, render_mode: str = None, sticky_actions_prob: float = 0.25):
    """
    Create and wrap the Atari environment.

    Applies a standard stack of wrappers for Atari experiments.
    """
    # NOTE: The v5 environments (e.g., ALE/Pong-v5) have sticky actions with p=0.25 by default.
    # To change this, we must pass the `repeat_action_probability` argument.
    env = gym.make(env_id, render_mode=render_mode, repeat_action_probability=sticky_actions_prob)

    # Seeding
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    # Episode stats (return/length in info["episode"])
    env = RecordEpisodeStatistics(env)

    # Atari preprocessing (grayscale, 84x84). Avoid double-skipping:
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=1,              # ALE/Pong-v5 already applies frameskip internally
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False,
    )

    # Reward clipping to [-1, 1]
    env = TransformReward(env, lambda r: float(np.clip(r, -1.0, 1.0)))

    # Stack 4 consecutive frames (output shape: (4, 84, 84))
    env = FrameStack(env, 4)

    return env