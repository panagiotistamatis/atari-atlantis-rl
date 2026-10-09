"""Proximal Policy Optimization (PPO) for Atari Atlantis.

Reuses the A2C backbone (`ActorCritic`, vectorised envs, GAE, stochastic
evaluation) and adds multiple epochs and minibatches per rollout plus the
clipped surrogate objective (Schulman et al. 2017), which keeps pi_new/pi_old
inside [1-eps, 1+eps] and so needs the old log-probs stored at rollout time.

The reward transform is applied in the loop so reported returns stay raw, and
GAE uses correct terminal masking. Defaults follow the CleanRL `ppo_atari`
reference.
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
    p = argparse.ArgumentParser(description="PPO for Atari Atlantis")
    p.add_argument("--exp-name", type=str, default="ppo")
    p.add_argument("--env-id", type=str, default="ALE/Atlantis-v5")
    p.add_argument("--total-timesteps", type=int, default=10_000_000)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=128,
                   help="Rollout length per environment before each update.")
    p.add_argument("--learning-rate", type=float, default=2.5e-4)
    add_bool_flag(p, "anneal-lr", True)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--clip-coef", type=float, default=0.1,
                   help="PPO surrogate clipping coefficient (epsilon).")
    add_bool_flag(p, "clip-vloss", True, "Use a clipped value-function loss.")
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    add_bool_flag(p, "norm-adv", True, "Per-minibatch advantage normalisation.")
    p.add_argument("--target-kl", type=float, default=None,
                   help="Optional approximate-KL early-stop threshold.")
    p.add_argument("--reward-mode", choices=["clip", "scale", "none"], default="clip")
    p.add_argument("--reward-scale", type=float, default=0.01)
    add_bool_flag(p, "fire-only", False)
    # [PopArt] optional value normalisation; use together with --reward-mode none.
    add_bool_flag(p, "popart", False,
                  "[PopArt] Pop-Art value normalisation (pair with --reward-mode none).")
    p.add_argument("--popart-beta", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--torch-deterministic", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--eval-every", type=int, default=500_000)
    p.add_argument("--eval-episodes", type=int, default=10)
    p.add_argument("--save-dir", type=str, default="runs")
    return p.parse_args()


def main():
    args = parse_args()
    batch_size = args.num_envs * args.num_steps
    minibatch_size = batch_size // args.num_minibatches
    num_iterations = args.total_timesteps // batch_size
    run_name = f"atlantis__{args.exp_name}__seed{args.seed}__{int(time.time())}"
    out_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed, args.torch_deterministic)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    print(f"Run: {run_name}\nDevice: {device}\nIterations: {num_iterations}  "
          f"batch: {batch_size}  minibatch: {minibatch_size}  reward_mode: {args.reward_mode}")

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

    # [PopArt] bind an optional normaliser to the value head. None => plain PPO.
    popart = None
    if args.popart:
        from .popart import PopArt
        popart = PopArt(agent.critic, device, beta=args.popart_beta)
        print(f"[PopArt] enabled (beta={args.popart_beta}); reward_mode={args.reward_mode}")

    # Rollout storage (obs kept as uint8).
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

    for iteration in range(1, num_iterations + 1):
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / num_iterations
            optimizer.param_groups[0]["lr"] = frac * args.learning_rate

        # ---- collect a rollout ------------------------------------------------
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                if popart is not None:          # [PopArt] store real-scale value for GAE
                    value = popart.denormalize(value)
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
            if popart is not None:              # [PopArt] real-scale bootstrap value
                next_value = popart.denormalize(next_value)
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

        # ---- flatten the batch -----------------------------------------------
        b_obs = obs.reshape((-1, *obs_shape))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # [PopArt] update running stats from this batch and rescale the value head
        # (POP step) BEFORE the epochs, so targets below are normalised consistently.
        if popart is not None:
            popart.update_and_preserve(b_returns)

        # ---- PPO update: several epochs over shuffled minibatches -------------
        b_inds = np.arange(batch_size)
        clipfracs = []
        approx_kl = torch.tensor(0.0)
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item())

                mb_adv = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy (surrogate) loss.
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - args.clip_coef,
                                                 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped).
                if popart is not None:          # [PopArt] train the normalised head
                    # newvalue is the raw head output (normalised space); the target
                    # is the real-scale return mapped into that same space.
                    v_loss = 0.5 * ((newvalue - popart.normalize(b_returns[mb_inds])) ** 2).mean()
                elif args.clip_vloss:
                    v_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss = 0.5 * torch.max(v_unclipped, v_loss_clipped).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # ---- logging ----------------------------------------------------------
        if writer:
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("charts/learning_rate",
                              optimizer.param_groups[0]["lr"], global_step)
        if iteration % 5 == 0:
            sps = int(global_step / (time.time() - start_time))
            avg_ret = np.mean(return_window) if return_window else float("nan")
            print(f"iter {iteration}/{num_iterations}  step {global_step}  "
                  f"avg_return(100)={avg_ret:.0f}  entropy={entropy_loss.item():.3f}  "
                  f"approx_kl={approx_kl.item():.4f}  SPS={sps}")

        # ---- periodic stochastic evaluation + checkpoint ---------------------
        if global_step >= next_eval_at:
            next_eval_at += args.eval_every
            agent.eval()
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
