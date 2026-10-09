"""Evaluate a trained agent, or measure the random baseline.

A random policy already scores ~20,000 on Atlantis, so the random baseline is
the bar a trained agent must clear, not zero.

Examples:
    python -m atari_rl.evaluate --algo random --episodes 30
    python -m atari_rl.evaluate --algo ppo --checkpoint runs/ppo/model.pt --record
    python -m atari_rl.evaluate --algo ppo --checkpoint runs/ppo/model.pt --gif --max-steps 400
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from .envs import make_env
from .models import ActorCritic


def run_episodes(action_fn, n_episodes, env_id, seed, fire_only, render_mode=None,
                 video_folder=None, max_steps=None, gif_path=None, gif_every=2,
                 gif_skip=0):
    """Run ``n_episodes`` and return the raw (true game score) episode returns.

    ``action_fn`` maps an observation to an integer action. ``max_steps`` caps
    each episode (e.g. for short videos); a capped episode reports the running
    raw-reward sum instead of the episode statistic. ``gif_path`` saves a GIF of
    the rendered frames: it keeps every ``gif_every``-th frame after the first
    ``gif_skip`` steps (to skip the quiet opening). Pair with ``max_steps``.
    """
    import gymnasium as gym

    env = make_env(env_id, seed, 0, fire_only, record_stats=True,
                   render_mode=render_mode)()
    if video_folder is not None:
        env = gym.wrappers.RecordVideo(env, video_folder,
                                       episode_trigger=lambda e: True)

    returns, frames = [], []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        ep_return, steps = 0.0, 0
        while not done:
            obs, reward, terminated, truncated, info = env.step(action_fn(obs))
            ep_return += float(reward)
            steps += 1
            done = terminated or truncated
            if gif_path is not None and steps > gif_skip and steps % gif_every == 0:
                frames.append(env.render())
            if max_steps is not None and steps >= max_steps:
                break
        # Natural end: RecordEpisodeStatistics value; capped by max_steps: running sum.
        returns.append(float(info["episode"]["r"]) if "episode" in info else ep_return)
    env.close()

    if gif_path is not None and frames:
        import imageio.v2 as imageio
        imageio.mimsave(gif_path, frames, fps=20)
        print(f"Saved {gif_path} ({len(frames)} frames)")
    return returns


def _model_action_fn(model, device, stochastic=False):
    """Action function for a loaded ActorCritic.

    Firing in Atlantis is edge-triggered: a deterministic policy repeats one
    action, holds fire down and shoots once, collapsing to the passive ~2,000
    score. So agents are evaluated by sampling from the policy
    (``stochastic=True``), which measures the policy actually learned.
    """

    def action_fn(obs):
        obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.uint8, device=device)
        with torch.no_grad():
            out = model(obs_t.unsqueeze(0))
            logits = out[0] if isinstance(out, tuple) else out
            if stochastic:
                probs = torch.softmax(logits, dim=1)
                return int(torch.multinomial(probs, 1).item())
            return int(torch.argmax(logits, dim=1).item())

    return action_fn


def main():
    parser = argparse.ArgumentParser(description="Evaluate an Atlantis agent")
    parser.add_argument("--algo", choices=["a2c", "ppo", "random"], required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--env-id", type=str, default="ALE/Atlantis-v5")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fire-only", action="store_true",
                        help="Restrict to fire actions (drops NOOP). Not recommended "
                             "for Atlantis: it prevents repeated firing.")
    parser.add_argument("--record", action="store_true",
                        help="Save gameplay videos to <checkpoint dir>/videos.")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Cap steps per episode (e.g. for short video clips).")
    parser.add_argument("--gif", action="store_true",
                        help="Save a GIF of the gameplay (pair with --max-steps).")
    parser.add_argument("--gif-out", type=str, default=None,
                        help="GIF output path (default: demo.gif next to the checkpoint).")
    parser.add_argument("--gif-skip", type=int, default=0,
                        help="Skip this many steps before capturing (skips the quiet opening).")
    parser.add_argument("--gif-every", type=int, default=2,
                        help="Keep every Nth frame in the GIF (higher = shorter/smaller).")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    fire_only = args.fire_only
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    render_mode = "rgb_array" if (args.record or args.gif) else None
    base = os.path.dirname(args.checkpoint) if args.checkpoint else "."
    video_folder = os.path.join(base, "videos") if args.record else None
    gif_path = None
    if args.gif:
        gif_path = args.gif_out or os.path.join(base, "demo.gif")

    if args.algo == "random":
        # Temporary env, only to get the action space.
        probe = make_env(args.env_id, args.seed, 0, fire_only, record_stats=False)()
        action_space = probe.action_space
        probe.close()
        rng = np.random.default_rng(args.seed)
        action_fn = lambda obs: int(rng.integers(action_space.n))  # noqa: E731
    else:
        if not args.checkpoint:
            parser.error("--checkpoint is required for --algo a2c/ppo")
        probe = make_env(args.env_id, args.seed, 0, fire_only, record_stats=False)()
        n_actions = probe.action_space.n
        probe.close()
        model = ActorCritic(n_actions).to(device)
        model.load_state_dict(torch.load(args.checkpoint, map_location=device,
                                         weights_only=True))
        model.eval()
        # Sample from the policy; never pure argmax (see _model_action_fn).
        action_fn = _model_action_fn(model, device, stochastic=True)

    returns = run_episodes(action_fn, args.episodes, args.env_id, args.seed,
                           fire_only, render_mode, video_folder, args.max_steps,
                           gif_path, args.gif_every, args.gif_skip)
    returns = np.array(returns)
    print(f"algo={args.algo} episodes={args.episodes}")
    print(f"  per-episode: {np.round(returns, 1).tolist()}")
    print(f"  mean={returns.mean():.1f}  std={returns.std():.1f}  "
          f"min={returns.min():.1f}  max={returns.max():.1f}")


if __name__ == "__main__":
    main()
