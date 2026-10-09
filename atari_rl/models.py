"""Actor-critic network for the A2C and PPO agents.

Uses the DeepMind "Nature" CNN torso (Mnih et al. 2015) with separate policy and
value heads. Weights use orthogonal init with the gains from CleanRL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
    """Orthogonal weight init with a configurable gain (CleanRL convention)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class NatureCNN(nn.Module):
    """Convolutional torso mapping (N, 4, 84, 84) uint8 frames to 512 features."""

    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pixels arrive as uint8 in [0, 255]; scale into [0, 1] on-device.
        return self.network(x.float() / 255.0)


class ActorCritic(nn.Module):
    """Shared-torso actor-critic (policy head gain 0.01 -> near-uniform initial policy)."""

    def __init__(self, n_actions: int, in_channels: int = 4):
        super().__init__()
        self.torso = NatureCNN(in_channels)
        self.actor = layer_init(nn.Linear(512, n_actions), std=0.01)
        self.critic = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x: torch.Tensor):
        hidden = self.torso(x)
        return self.actor(hidden), self.critic(hidden)

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        # squeeze(-1): shape (N,); else (N, 1) broadcasts against (N,) into (N, N) in GAE.
        return self.critic(self.torso(x)).squeeze(-1)

    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or evaluate) an action and return log-prob, entropy, value."""
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)
