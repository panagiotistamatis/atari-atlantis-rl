# Technical report: Deep RL on Atari Atlantis

A detailed write-up of what I built, what I changed and why, and what the results
mean. It is written as a reference I can speak from, for example in an interview.
The short version lives in the README; this is the deep dive.

## 1. Summary

I built a reinforcement-learning pipeline that learns to play the Atari 2600 game
Atlantis from raw pixels. The final agent uses PPO and reaches about 820,000 points
in stochastic evaluation, roughly 40 times a random policy (about 20,000), and at
its plateau it survives the full episode (the Atari time limit of about 27,000
steps) every time. The project started from a coursework version that did not learn
at all, and most of the value is in the chain of concrete diagnoses and fixes that
turned it into a working agent, plus a set of controlled experiments on reward
representation, including honest negative results.

## 2. The problem

Atlantis is a fixed-gun shooter. The player controls three turrets and only chooses
which one fires. The observation is four stacked 84x84 grayscale frames (the stack
gives the network motion information that a single frame cannot). The action set is
NOOP, FIRE, RIGHTFIRE, LEFTFIRE. Episodes are long (thousands of steps), and the
real skill is defensive: destroy the right incoming ships so the city survives
longer, because the score grows with survival time.

Two properties make Atlantis an interesting test. First, a random policy already
scores highly (about 20,000), because firing across the three guns destroys many
attackers by chance, so the meaningful question is not "does it score" but "does it
beat random". Second, firing is edge-triggered in the emulator, which caused two
separate failures described below.

## 3. Architecture and algorithms

Both agents share the DeepMind "Nature" CNN torso (three convolutional layers then
a 512-unit dense layer) with separate policy and value heads, orthogonal weight
initialization, and pixel normalization inside the network. On top of that shared
backbone:

- **A2C** collects a short rollout from several parallel environments, estimates
  advantages with GAE, and does a single full-batch gradient step. It is the
  baseline.
- **PPO** adds three things to A2C: several optimization epochs over each rollout,
  minibatches within each epoch, and a clipped surrogate objective that keeps the
  policy ratio close to one so reusing the data cannot move the policy too far. This
  makes it more sample-efficient and more stable. It is the final agent.

The implementation follows the structure of the CleanRL reference agents. GAE, the
reward handling, seeding, and the evaluation path are shared between the two.

## 4. The debugging journey

This is the core of the project. Each item is a problem I found and the fix.

**4.1 A broken advantage target.** The original training loop summed a single step's
reward but bootstrapped with `gamma**n * V(s')`, so what it called an n-step return
was actually a mis-discounted, biased target. I replaced it with Generalized
Advantage Estimation, GAE(lambda), with correct masking at episode boundaries. While
wiring this in I hit a tensor-shape bug: the value head returned shape `(N, 1)` while
the stored values were `(N,)`, which silently broadcast into an `(N, N)` matrix
inside the GAE recursion and crashed. Squeezing the value to `(N,)` fixed it. This is
a good example of how a shape mismatch in RL does not always error cleanly; it can
corrupt the math.

**4.2 Reward magnitude swamped the value function.** Atlantis rewards are large,
hundreds per event and hundreds of thousands per episode. If those go straight into
the critic's mean-squared-error loss, the value loss dwarfs the policy and entropy
gradients, and because the value head shares the convolutional trunk, the actor
barely updates. The standard fix is to sign-clip the reward used for learning (every
scoring event becomes +1), which keeps the value targets well-scaled. Importantly, I
report the raw, unclipped game score for every result, so the numbers stay
comparable to the random baseline. The reward transform is applied in the training
loop, not inside the environment, precisely so the logged return is always the true
score.

**4.3 Firing is edge-triggered, so dropping NOOP was actively harmful.** Early on I
"optimized" the action space by removing NOOP and keeping only the three FIRE
actions, reasoning that the player only ever fires. This made the agent worse, not
better. The gun fires only on the transition from released to pressed, so a policy
that sends a FIRE action every step holds the button down and fires exactly once. I
confirmed this with a small probe (`diagnose.py`) that runs fixed policies: every
constant action (including always-FIRE) scores a flat 2,000, which is just the passive
wave-completion bonus, and dies in about 420 steps, while a random policy that
interleaves NOOP scores about 16,840 over about 1,300 steps. The fix was to keep the
full action set. The lesson is that a plausible domain "optimization" can silently
break the environment's own mechanics.

**4.4 Deterministic evaluation collapses.** For the same edge-triggered reason, a
greedy (argmax) evaluation policy holds one button and falls back to the passive
2,000, even when the trained policy is good. So I evaluate by sampling from the
policy, which measures what the agent actually learned. This is legitimate for a
policy-gradient method: the stochastic policy is the policy.

**4.5 A2C learns but slowly; PPO masters the game.** With all of the above fixed,
A2C climbs above random but only reaches about 58,000 after 3M steps, and its
training curve is noisy. PPO, reusing the exact same network and environment,
reaches about 590,000 at 1M steps and plateaus near 820,000. The gap is the expected
one: PPO's data reuse and clipping make it far more sample-efficient and stable on
this task.

## 5. Reward-representation experiments

Once the agent was surviving the entire episode, the only remaining headroom was
scoring efficiency (at a fixed episode length the score still varied by roughly 15
percent between episodes). Sign-clipping throws away reward magnitude: a big command
ship worth thousands counts the same +1 as a small kill. So I tested two ways to use
the true magnitudes, as controlled A/B runs against the clip baseline.

- **Plain scaling** (raw reward times 0.01) failed. It stayed at random level,
  because the unclipped return targets are large and high-variance, the critic cannot
  fit them, and the advantages become too noisy to drive the policy. This is the same
  value-scale problem as 4.2, now showing up in the targets.
- **Pop-Art** value normalization (van Hasselt et al. 2016) is the principled fix.
  It trains the value head on targets normalized to zero mean and unit variance,
  while rescaling the head's output layer whenever the running statistics change so
  that the denormalized predictions are preserved. I implemented it as an isolated,
  off-by-default feature (one file, `popart.py`, plus a handful of tagged hooks). It
  does learn, which confirms the implementation, but about ten times slower than
  sign-clipping on this game, so it did not beat the baseline.

Conclusion: sign-clipping was the right choice for Atlantis, and I kept both
experiments as documented negative results rather than hiding them. Being able to
explain why scaling fails and why Pop-Art is the correct response is the point.

## 6. Results

- Random policy: about 20,000 (30 episodes).
- A2C at 3M steps: about 58,000, roughly 3x random.
- PPO at 10M steps: about 820,000, roughly 40x random, plateauing from about 5M to 6M
  steps onward. At the plateau every episode reaches the Atari time limit of about
  27,000 steps, meaning the agent stops dying and plays the whole level. Within this
  setup that is effectively solved.

For context, the published literature reaches into the millions on Atlantis with far
more training and tuning, so there is headroom, but the goal here was a clean, honest,
demonstrably-learning agent rather than a leaderboard number, and the time-capped
plateau shows the agent has learned to survive rather than just to score.

## 7. Engineering and reproducibility

- Everything is seeded (Python, NumPy, PyTorch, and the environments) and driven by
  argparse, with no hyperparameters buried as module constants.
- The environment stack is pinned to a tested combination (`gymnasium==0.29.1`,
  `ale-py==0.8.1`) and the training loops assume that version's vector-env autoreset
  and episode-statistics behavior. A small compatibility shim covers the frame-stack
  wrapper rename across Gymnasium versions.
- Observations are stored as uint8 and normalized on the GPU to save memory.
- Each run logs per-episode returns and periodic evaluation scores to CSV, plus a
  TensorBoard log, so the figures are reproducible from the logs.
- A practical but important detail: training was accidentally running on CPU because
  the installed PyTorch was the CPU-only build. Installing the CUDA build on the
  available laptop GPU cut a 10M PPO run from impractical to about six hours.

## 8. Limitations and next steps

- Single seed per configuration. A portfolio-grade claim would average several seeds
  with confidence intervals.
- The CPU-side environment stepping (a synchronous vector env) is the throughput
  bottleneck, not the GPU. An asynchronous vector env or more parallel environments
  would speed things up.
- The score plateau is partly a function of the episode time limit. Raising the limit
  would raise the raw number without necessarily meaning better play.
- Pop-Art could likely be made competitive with more tuning (its beta rate, and a
  larger value-loss weight), but it was not worth it once sign-clipping had solved the
  game.

## 9. Interview talking points

- I can describe the full actor-critic stack from pixels to policy, and explain GAE,
  the PPO clipped objective, and why PPO beats A2C here.
- The strongest story is the debugging chain: a mis-discounted target and a shape bug,
  the value-scale problem, and in particular the two edge-triggered firing findings
  (dropping NOOP breaks firing; deterministic evaluation collapses), which I found
  with a purpose-built probe rather than by guessing.
- I can argue the reward-representation trade-off from first principles and back it
  with my own A/B results, including why plain scaling fails and why Pop-Art is the
  principled alternative.
- I value honest negative results: the repository keeps the experiments that did not
  win, with the reasons.
