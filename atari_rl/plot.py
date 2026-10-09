"""Plot learning curves from one or more training runs.

Reads the ``episodes.csv`` written by the trainers and plots smoothed episodic
return against environment steps, with an optional random-baseline line.

Example:
    python -m atari_rl.plot --runs ppo=runs/atlantis__ppo__seed1__123 \
        a2c=runs/atlantis__a2c__seed1__456 --random 20000 --out assets/learning_curve.png
"""

from __future__ import annotations

import argparse
import csv
import os


def _read_episodes(run_dir):
    steps, returns = [], []
    with open(os.path.join(run_dir, "episodes.csv")) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["global_step"]))
            returns.append(float(row["episodic_return"]))
    return steps, returns


def _rolling_mean(values, window):
    from collections import deque
    out = []
    buf = deque(maxlen=window)
    for v in values:
        buf.append(v)
        out.append(sum(buf) / len(buf))
    return out


def main():
    p = argparse.ArgumentParser(description="Plot learning curves")
    p.add_argument("--runs", nargs="+", required=True,
                   help="label=run_dir pairs, e.g. a2c=runs/atlantis__a2c__...")
    p.add_argument("--random", type=float, default=None,
                   help="Random-baseline score to draw as a reference line.")
    p.add_argument("--window", type=int, default=50, help="Rolling-mean window (episodes).")
    p.add_argument("--out", type=str, default="assets/learning_curve.png")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    plt.figure(figsize=(8, 5))

    for item in args.runs:
        label, run_dir = item.split("=", 1)
        steps, returns = _read_episodes(run_dir)
        plt.plot(steps, _rolling_mean(returns, args.window), label=label)

    if args.random is not None:
        plt.axhline(args.random, linestyle="--", color="grey",
                    label=f"random baseline ({args.random:.0f})")

    plt.xlabel("Environment steps")
    plt.ylabel(f"Episodic return (rolling mean, {args.window})")
    plt.title("Atari Atlantis -- learning curves")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
