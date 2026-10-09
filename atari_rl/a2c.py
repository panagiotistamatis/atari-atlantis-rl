"""Synchronous Advantage Actor-Critic (A2C) for Atari Atlantis.

N parallel environments produce a fixed-length rollout, advantages are estimated
with GAE(lambda), and the policy/value/entropy losses take one full-batch
gradient step per rollout. This is the baseline for the PPO agent in ppo.py.

Design notes:

* The reward transform (default sign-clipping) is applied to the learning signal
  only; reported episode returns are the raw game score, directly comparable to
  the random baseline (~20,000 on Atlantis).
* GAE(lambda) with correct terminal masking.
* Advantage normalisation is OFF by default: with tiny advantages it amplifies
  noise and drove entropy collapse.

Credit: the loop structure follows the CleanRL reference implementations; the
value baseline follows Mnih et al. (2016, A3C/A2C).
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .envs import iter_episode_stats, make_vector_env, transform_reward
from .evaluate import _model_action_fn, run_episodes
from .models import ActorCritic
from .utils import CSVLogger, add_bool_flag, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="A2C for Atari Atlantis")
    p.add_argument("--exp-name", type=str, default="a2c")
    p.add_argument("--env-id", type=str, default="ALE/Atlantis-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=16,
                   help="Rollout length per environment before each update.")
    p.add_argument("--learning-rate", type=float, default=7e-4)
    add_bool_flag(p, "anneal-lr", True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    add_bool_flag(p, "norm-adv", False,
                  "Normalise advantages per batch (off by default).")
    p.add_argument("--reward-mode", choices=["clip", "scale", "none"], default="clip")
    p.add_argument("--reward-scale", type=float, default=0.01)
    # fire-only OFF by default: without NOOP the fire button stays held and the
    # gun fires only once (firing is edge-triggered).
    add_bool_flag(p, "fire-only", False)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--save-dir", type=str, default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    num_updates = args.total_timesteps // batch_size
    run_name = f"atlantis__{args.exp_name}__seed{args.seed}__{int(time.time())}"
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name}\nDevice: {device}\nUpdates: {num_updates}  "
          f"batch_size: {batch_size}  reward_mode: {args.reward_mode}")

    # TensorBoard is optional; the CSV logs are the source of truth for plots.
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out_dir)
    except Exception:  # pragma: no cover
        writer = None

    ep_logger = CSVLogger(os.path.join(out_dir, "episodes.csv"),
                          ["global_step", "episodic_return", "episodic_length"])
    eval_logger = CSVLogger(os.path.join(out_dir, "evals.csv"),
                            ["global_step", "mean_return", "std_return"])

    envs = make_vector_env(args.env_id, args.num_envs, args.seed, args.fire_only)
    n_actions = envs.single_action_space.n
    obs_shape = envs.single_observation_space.shape

    agent = ActorCritic(n_actions).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage (obs kept as uint8; scaled inside the net).
    obs = torch.zeros((args.num_steps, args.num_envs, *obs_shape),
                      dtype=torch.uint8, device=device)
    actions = torch.zeros((args.num_steps, args.num_envs), dtype=torch.long, device=device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones = torch.zeros((args.num_steps, args.num_envs), device=device)
    values = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    start_time = time.time()
    return_window = deque(maxlen=100)
    next_eval_at = args.eval_every

    reset_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.as_tensor(reset_obs, dtype=torch.uint8, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- collect a rollout ------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
            actions[step] = action
            logprobs[step] = logprob
            values[step] = value

            next_obs_np, reward_raw, terminated, truncated, infos = envs.step(
                action.cpu().numpy())
            reward_train = transform_reward(np.asarray(reward_raw, dtype=np.float32),
                                            args.reward_mode, args.reward_scale)
            rewards[step] = torch.as_tensor(reward_train, device=device)
            next_done = torch.as_tensor(
                np.logical_or(terminated, truncated).astype(np.float32), device=device)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.uint8, device=device)

            for ret, length in iter_episode_stats(infos, args.num_envs):
                return_window.append(ret)
                ep_logger.log(global_step, ret, length)
                if writer:
                    writer.add_scalar("charts/episodic_return", ret, global_step)

        # ---- GAE(lambda) ------------------------------------------------------
        with torch.no_grad():
            next_value = agent.get_value(next_obs)
            advantages = torch.zeros_like(rewards)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam)
            returns = advantages + values

        # ---- flatten and do a single A2C update ------------------------------
        b_obs = obs.reshape((-1, *obs_shape))
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs, b_actions)
        if args.norm_adv:
            b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        pg_loss = -(b_advantages * newlogprob).mean()
        v_loss = 0.5 * ((newvalue - b_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
        optimizer.step()

        # ---- logging ----------------------------------------------------------
        if writer:
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
        if update % 20 == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_ret = np.mean(return_window) if return_window else float("nan")
            print(f"update {update}/{num_updates}  step {global_step}  "
                  f"avg_return(100)={avg_ret:.0f}  entropy={entropy_loss.item():.3f}  "
                  f"SPS={sps}")

        # ---- periodic stochastic evaluation + checkpoint ---------------------
        if global_step >= next_eval_at:
            next_eval_at += args.eval_every
            agent.eval()
            # Sample from the policy (not argmax): a deterministic policy holds fire
            # and collapses to the passive score.
            eval_returns = run_episodes(
                _model_action_fn(agent, device, stochastic=True),
                args.eval_episodes, args.env_id, args.seed + 10_000, args.fire_only)
            agent.train()
            m, s = float(np.mean(eval_returns)), float(np.std(eval_returns))
            eval_logger.log(global_step, m, s)
            if writer:
                writer.add_scalar("eval/mean_return", m, global_step)
            print(f"  [eval @ {global_step}] stochastic mean={m:.0f} +/- {s:.0f}")
            torch.save(agent.state_dict(), os.path.join(out_dir, "model.pt"))

    torch.save(agent.state_dict(), os.path.join(out_dir, "model.pt"))
    envs.close()
    if writer:
        writer.close()
    print(f"Done. Artifacts in {out_dir}")


if __name__ == "__main__":
    main()
