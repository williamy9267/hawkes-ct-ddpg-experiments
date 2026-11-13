# Power-law Hawkes CT-DDPG experiment

The true simulator uses the classical history-based power-law kernel

```text
phi(u) = b * a^b * (u + a)^(-(1 + b)),
lambda_t = mu(t, X_t, a_t) + alpha q(a_t) int phi(t-s) dN_s,
```

with `a=0.12`, `b=0.80`, and no global link function. Because this kernel has
no finite exact Markovization, filtered policies use the arithmetic dictionary
`beta_k=k`, `k=1,...,20`. The DGM/HJB oracle knows the globally fitted
signed mixture and all model parameters. Evaluation for every policy still
uses the true history-based simulator.

## Maintained comparison

CT-DDPG, SAC, and DDPG are reported under two observation protocols:

- `current`: `(t, X)` only.
- `filters`: `(t, X, Z1, ..., Z20)` using normalized observed filters.

The learning agents do not receive fitted mixture weights. Power-law filtered
SAC/DDPG warm-start from their matching current-state policies with neutral
weights on the added inputs, then train and validate independently. History
windows and PPO are omitted from the maintained notebook.
Older history-observation artifacts are preserved under
`archived_history_observation` and ignored by the active suite.

The uncapped clustered calibration uses `T=8`, `mu0=0.60`, excitation `0.99`,
jump amplitude `clip(0.50 + 0.002 X - 0.05 a, 1e-4, 0.90)`, and cost
coefficients `(c_x, c_a, c_N, c_T)=(1.00, 0.50, 0, 0.50)`. Jumps remain
positive; no prehistory or marked-jump device is used. The true-kernel
finite-horizon offspring mass is `0.95601 < 1`, and the fitted positive-link
envelope has mass `0.93837 < 1`. The arithmetic dictionary has about `0.7%`
relative L2 kernel error on `[0,T]`.

CT-DDPG filtered training is sequential: it is warm-started from the matching
trained current-state checkpoint, with a centered generic filter response that
uses only the preselected decay grid. On an independent 800-path seed-999
comparison, the costs are:

```text
known-parameter DGM/HJB  1.5696 +/- 0.0617
CT-DDPG filtered        1.5870 +/- 0.0607
CT-DDPG current         1.6647 +/- 0.0634
SAC filtered            1.7056 +/- 0.0641
DDPG filtered           1.7410 +/- 0.0680
DDPG current            1.7669 +/- 0.0644
SAC current             1.7920 +/- 0.0665
static a=0.34           1.9242 +/- 0.0734
```

Filtered CT-DDPG improves `4.67%` over its current-state ablation on seed 999
and `3.63%` on a second independent 800-path seed. Its mean action is lower
but its action variation is higher, consistent with selective intervention
during clusters. It is the best model-free learned policy in the common-path
comparison and lies close to the privileged DGM/HJB oracle. DDPG benefits
from filtering and remains behind CT-DDPG. Filtered SAC improves
4.82% over SAC current on seed 999 and remains better on two additional
independent 800-path blocks. Checkpoints are compared on a reset common-random
validation block, and filtered SAC runs for the full 200,000-transition budget
before restoring its best iterate. The stabilized DDPG pair uses smaller critic
targets, a larger replay batch, delayed learning, and a disclosed constant-action
initialization. Filtering improves DDPG by 1.47% on seed 999, with the same
direction on two additional 800-path blocks.

The fully trained filtered policies use the same 20 observed filter states.
CT-DDPG uses 72,000 environment transitions, compared with 200,000 for SAC and
160,000 for DDPG. The separate budget figure reports final common-test cost
against these complete training budgets; it does not shorten or weaken either
baseline to enlarge CT-DDPG's lead.

## Execution

The maintained notebook is:

```text
powerlaw_kernel_experiment/powerlaw_kernel_comparison.ipynb
```

Cached checkpoints load by default. The tuning cell contains separate force
switches for CT-DDPG, DGM, and SB3. Standalone training commands are:

```powershell
python train_tuned_ct_powerlaw_suite.py --out-dir results --force --max-workers 1
python train_tuned_sb3_powerlaw_suite.py --out-dir results --force --max-workers 4
```

SB3 uses four CPU rollout environments and continuous-time discount
`exp(-beta*dt)`. DGM uses CUDA when available. Important outputs are
`results/comparison.json`, `figures/benchmark_costs.png`,
`results/sac_filter_fix_validation.json`,
`figures/current_vs_filtered_costs_expanded.png`, convergence PNGs, `figures/filtered_cost_vs_training_budget.png`,
and value-surface PNGs, `figures/policy_path_ensemble_ci.png`, and
`figures/ct_kernel_approximation.png`.
