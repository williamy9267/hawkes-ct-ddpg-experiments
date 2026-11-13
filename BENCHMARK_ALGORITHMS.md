# Benchmark Algorithms and Comparability Notes

This note describes the maintained non-oracle benchmarks beside filtered
CT-DDPG in the Erlang and power-law Hawkes experiments. The reported generic
RL methods are SAC and DDPG. PPO and event-history-window variants remain only
in archival utilities and are not part of the active notebooks.

## Common evaluation protocol

Every reported policy uses:

- the same true Hawkes jump-diffusion simulator within an experiment;
- scalar physical control `a in [0,1]`;
- the same running and terminal objective;
- deterministic actions during evaluation;
- continuous-time discount `exp(-beta*t)`;
- common path seeds and the same number of Monte Carlo episodes;
- unscaled evaluation returns, even when SB3 uses training-only reward scaling.

There is no event-count cost and no cap on total intensity. Each filtered
policy generates one closed-loop path: observed jumps update its filters, its
filter-based action then controls the subsequent observed system, and no
counterfactual policy with access to unavailable exact memory is simulated.

The plotted quantity is positive discounted cost, so lower is better. Reported
90% intervals quantify rollout uncertainty conditional on one trained policy.
They do not include training-seed uncertainty.

## Observation protocols

The maintained comparison has two observation modes for CT-DDPG, SAC, and
DDPG.

`current`:

- normalized time;
- current diffusion state `X_t`;
- no explicit summary of Hawkes memory.

This is a deliberately memory-blind ablation. The problem is generally not
Markovian in `(t,X_t)`, so it should not be presented as solving the fully
observed control problem.

`filters`:

- normalized time;
- current state `X_t`;
- the full preselected exponential filter bank generated online from observed
  jumps.

The base Erlang filter observation has dimension 14 and the power-law
observation has dimension 22. Erlang SAC deterministically augments the 12
filter levels with 11 adjacent decay-rate differences, giving network input
dimension 25. These finite differences approximate
\(-\beta\,\partial_\beta Z(\beta)\), an age-weighted memory coordinate, using
only the same observed bank and preselected decay grid. The learning agents
receive neither the true Erlang phases, the true kernel parameters, nor fitted
power-law mixture weights. Thus all filtered methods have the same stochastic
information, although their numerical feature conditioning may differ, while
the oracle keeps its privileged known-model information.

This two-mode design answers two direct questions:

1. Does an explicit approximate Markov state improve a given learning
   algorithm over current state alone?
2. Given the same filtered state, does the continuous-time CT-DDPG objective
   improve on generic discrete-time actor-critic training?

## SAC

SAC is a generic off-policy, entropy-regularized continuous-control baseline.
It is implemented with Stable-Baselines3 and an MLP policy. Training is
stochastic, but evaluation uses `deterministic=True`.

SAC does not use the CT-DDPG continuous-time value/advantage-rate identities.
Its filtered variant is generated from the same observed filter bank as
CT-DDPG. In the Erlang experiment, generic adjacent-rate features improve the
conditioning of the highly collinear bank; in the power-law experiment SAC
uses all normalized levels directly. In both experiments it warm-starts from
SAC-current before learning dependence on the added memory inputs.

## DDPG

DDPG is an off-policy deterministic actor-critic baseline with a Q critic,
target networks, replay, and Gaussian exploration noise. It is also implemented
with Stable-Baselines3.

DDPG is the closest generic baseline conceptually, but it still optimizes a
discrete-time Q-learning objective rather than the continuous-time
value/advantage-rate system. The filtered variant warm-starts from
DDPG-current in both memory experiments.

## CT-DDPG

The proposed method uses:

- a deterministic actor;
- a value network and target value network;
- an advantage-rate (`qbar`) network;
- replayed continuous-time path segments;
- Polyak target updates;
- observed exponential memory filters in the proposed variant;
- for the accepted Erlang calibration, a zero-initialized three-coefficient
  filter-rate profile residual learned by the same CT actor gradient.

`CT-DDPG filtered` is the theory-aligned proposed method. `CT-DDPG current` is
the memory-blind ablation and does not inherit the Markov-state justification.
The actor and value target networks in this implementation are stabilization
devices; the actor target is used in bootstrapped target construction, while
the online actor supplies the policy gradient.

## Oracle and static benchmarks

The Erlang DGM/HJB oracle knows the exact finite Markov state
`(t,X,Y1,Y2)`. The power-law DGM/HJB oracle knows the signed exponential-mixture
approximation and solves on `(t,X,Z1,...,Z20)`. Its resulting policy is evaluated
on the true history-based power-law simulator.

The oracle is not model-free and is not an algorithmically equal competitor.
It is a known-parameter performance reference. The static actions are selected
on separate validation paths and fixed for final evaluation. The selected
actions are 0.26 for Erlang and 0.34 for power law.

## Training and checkpointing

The wrappers use separate checkpoint-validation paths and restore the best
validation iterate. Before each SB3 checkpoint comparison, the validation
environment is reset to one fixed seed block, so a later iterate is compared
with its warm start on common random paths. Final comparison paths are not used
for gradient updates.
SB3 uses four vectorized rollout environments. Reward scaling is training-only
and leaves the reported objective unchanged: the maintained runs use scale 25,
except stabilized power-law DDPG uses scale 10 to control critic-target tails.
Filtered SAC/DDPG warm-start from the corresponding current-state checkpoint
with neutral initial memory weights. CT-DDPG uses mode-specific actor step
sizes and Polyak rates selected before the final comparison seed. The Erlang
profile residual uses only observed filter coordinates and the preselected
rate grid; it receives neither fitted kernel weights nor oracle states. The
revised power-law filtered actor is trained sequentially from the current
CT-DDPG checkpoint and starts from a centered generic level/rate-slope profile.
That profile uses only observed filters and the preselected arithmetic grid,
not fitted power-law mixture weights.

Cached checkpoints load by default in both notebooks. Force switches are
separate for CT-DDPG, DGM, and SB3 so figures can be regenerated without
silently retraining every method.

## Interpretation of the current results

Erlang final-seed mean costs are approximately:

```text
DGM 0.1037 < CT-filtered 0.1056 < SAC-filtered 0.1065
           < SAC-current 0.1093 < DDPG-filtered 0.1103
           < CT-current 0.1219 < DDPG-current 0.1266 < static a=0.26 0.1314
```

The revised sharp, near-critical power-law calibration has the following
seed-999 results over 800 common paths:

```text
DGM 1.5696 < CT-filtered 1.5870 < CT-current 1.6647
           < SAC-filtered 1.7056 < DDPG-filtered 1.7410
           < DDPG-current 1.7669 < SAC-current 1.7920
           < static a=0.34 1.9242
```

Filtered CT-DDPG improves over current CT-DDPG by about 13.4% for Erlang and
by 4.67% and 3.63% on two independent 800-path power-law evaluations. The
revised power-law DGM/SAC/DDPG checkpoints use the same environment and
objective. Filtered SAC improves over current SAC by 2.55% for Erlang and
4.82% for power law on the displayed test blocks, with the direction confirmed
on two additional 800-path blocks for each experiment. Filtered CT-DDPG is the
best model-free learned policy; the
known-parameter DGM/HJB oracle remains slightly better. Because the confidence
intervals condition on one trained policy and overlap for some methods, formal
superiority claims require several independent training seeds.

Power-law DDPG now uses a stable critic scale, batch size 1024, 15,000-step
learning delay, and a disclosed physical-action initialization at 0.30. Its
filtered continuation improves on current DDPG by 1.47% on seed 999 and on two
additional 800-path blocks. This replaces the former tail-unstable current
checkpoint; it is not a plotting clamp or a reduced evaluation sample.

The filtered-state count is held fixed within each experiment: 12 Erlang
readouts or 20 power-law readouts for every filtered learner. Full training
budgets are 35k/220k/160k transitions for Erlang CT-DDPG/SAC/DDPG and
72k/200k/160k for power law. The budget figures therefore provide a fairer
visual statement of CT-DDPG's sample efficiency than deliberately undertraining
SAC or changing the information supplied to it.

## Reporting checklist

- State whether the learner received `current` or `filters` observations.
- Distinguish exact Erlang oracle state from generic observed filters.
- State that the power-law oracle is an oracle for a finite mixture
  approximation but is evaluated in the true-history environment.
- Report unscaled discounted cost and common-path Monte Carlo intervals.
- Report action mean and standard deviation to detect policy collapse.
- Separate rollout uncertainty from training-seed uncertainty.
- Do not describe a numerical single-seed lead as statistical significance.

A suitable caption is:

> SAC and DDPG are Stable-Baselines3 continuous-control baselines trained on
> the same Hawkes simulator and evaluated on the same unscaled discounted
> objective as CT-DDPG. For each algorithm, the current-state ablation observes
> only `(t,X_t)`, while the filtered variant receives the same preselected bank
> of online exponential jump filters. DGM/HJB is a privileged known-parameter
> oracle. Error bars are 90% Monte Carlo intervals conditional on one trained
> policy.
