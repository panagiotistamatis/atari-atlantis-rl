"""Environment probe: does the Atlantis score depend on the action?

Runs fixed policies (do-nothing, fire-each-gun, fully random) and prints score
and length statistics. If do-nothing scores the same as active firing, the
reward carries no learning signal ("passive scoring") and no agent can improve.

Run:
    python -m atari_rl.diagnose --episodes 5
"""

from __future__ import annotations

import argparse

import numpy as np

from .envs import make_env


def _run_fixed(label, pick_action, env_id, episodes, seed):
    env = make_env(env_id, seed, 0, fire_only=False, record_stats=True)()
    scores, lengths = [], []
    for ep in range(episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(pick_action(env))
            done = terminated or truncated
        scores.append(float(info["episode"]["r"]))
        lengths.append(int(info["episode"]["l"]))
    env.close()
    s, ln = np.array(scores), np.array(lengths)
    print(f"  {label:20s} score: mean={s.mean():8.1f} std={s.std():7.1f} "
          f"min={s.min():7.1f} max={s.max():7.1f}   len: mean={ln.mean():5.0f}")


def main():
    p = argparse.ArgumentParser(description="Probe whether Atlantis reward depends on action")
    p.add_argument("--env-id", type=str, default="ALE/Atlantis-v5")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    # Temporary env, only to read the full action meanings.
    probe = make_env(args.env_id, args.seed, 0, fire_only=False, record_stats=False)()
    meanings = probe.unwrapped.get_action_meanings()
    probe.close()
    print(f"Action meanings: {list(enumerate(meanings))}")
    idx = {name: i for i, name in enumerate(meanings)}

    print(f"\n{args.episodes} episodes per fixed policy "
          f"(same seeds across policies -> differences are caused by the action):\n")

    if "NOOP" in idx:
        _run_fixed("always NOOP", lambda e: idx["NOOP"], args.env_id, args.episodes, args.seed)
    for name in ("FIRE", "RIGHTFIRE", "LEFTFIRE"):
        if name in idx:
            _run_fixed(f"always {name}", lambda e, n=name: idx[n],
                       args.env_id, args.episodes, args.seed)
    rng = np.random.default_rng(args.seed)
    _run_fixed("random (full set)", lambda e: int(rng.integers(e.action_space.n)),
               args.env_id, args.episodes, args.seed)

    print("\nRead-out:")
    print("  - If 'always NOOP' scores the SAME as the firing policies, the reward")
    print("    does not depend on the action -> passive-scoring trap (must change setup).")
    print("  - If firing policies score HIGHER / with variance, the signal is there")
    print("    and the training setup just needs tuning to exploit it.")


if __name__ == "__main__":
    main()
