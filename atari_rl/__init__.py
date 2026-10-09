"""Reinforcement-learning agents (PPO, A2C) for Atari Atlantis.

Modules:
    envs      -- environment construction and Atari wrappers
    models    -- Nature-CNN actor-critic
    ppo       -- Proximal Policy Optimization trainer (final agent)
    a2c       -- synchronous Advantage Actor-Critic trainer (baseline)
    popart    -- optional Pop-Art value normalization
    diagnose  -- diagnostic tool
    evaluate  -- policy evaluation and the random baseline
    plot      -- learning-curve plotting
    utils     -- seeding and CSV logging
"""

__version__ = "1.0.0"
