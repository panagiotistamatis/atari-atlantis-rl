"""Pop-Art value normalization (van Hasselt et al. 2016). Optional add-on.

Enabled with ``--popart`` in ppo.py (every hook there is tagged ``# [PopArt]``).
With the flag off, PPO is unchanged; to remove the feature, delete this file and
the ``# [PopArt]`` lines in ppo.py.

The value head is trained on targets normalised to zero mean / unit variance,
and its output layer is rescaled whenever the running statistics change so the
denormalised predictions are preserved. This allows learning from unclipped
reward magnitudes without the value-loss blow-up of plain reward scaling.
"""

import torch


class PopArt:
    """Running normaliser bound to the critic's final linear layer.

    ``output_layer`` must be the ``nn.Linear(512, 1)`` value head whose weights
    are rescaled to preserve outputs. ``beta`` is the EMA rate for the statistics.
    """

    def __init__(self, output_layer, device, beta=3e-4, epsilon=1e-4):
        self.layer = output_layer
        self.beta = beta
        self.epsilon = epsilon
        self.mu = torch.zeros(1, device=device)
        self.nu = torch.ones(1, device=device)   # running E[R^2]
        self._initialized = False

    @property
    def sigma(self):
        return torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=self.epsilon))

    def normalize(self, x):
        """Map real-scale targets to the normalised space the head learns in."""
        return (x - self.mu) / self.sigma

    def denormalize(self, x):
        """Map the head's normalised output back to the real reward scale."""
        return x * self.sigma + self.mu

    @torch.no_grad()
    def update_and_preserve(self, returns):
        """Update running stats from this batch of returns and rescale the value
        head so that denormalised predictions are unchanged (the POP step)."""
        old_sigma = self.sigma.clone()
        old_mu = self.mu.clone()

        batch_mu = returns.mean()
        batch_nu = (returns ** 2).mean()
        if not self._initialized:
            # Seed from the first batch instead of crawling up from (0, 1).
            self.mu = batch_mu.reshape(1).clone()
            self.nu = batch_nu.reshape(1).clone()
            self._initialized = True
        else:
            self.mu = (1 - self.beta) * self.mu + self.beta * batch_mu
            self.nu = (1 - self.beta) * self.nu + self.beta * batch_nu

        new_sigma = self.sigma
        # Preserve outputs: V = sigma * (W h + b) + mu must be invariant, so
        #   W' = W * old_sigma / new_sigma
        #   b' = b * old_sigma / new_sigma + (old_mu - new_mu) / new_sigma
        self.layer.weight.mul_(old_sigma / new_sigma)
        self.layer.bias.mul_(old_sigma / new_sigma)
        self.layer.bias.add_((old_mu - self.mu) / new_sigma)
