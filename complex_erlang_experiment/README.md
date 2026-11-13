# Erlang-kernel Hawkes CT-DDPG experiment

This experiment uses a classical additive Erlang-2 Hawkes intensity,

```text
lambda_t = mu(t, X_t, a_t) + alpha q(a_t) Y2_t,
```

with no global link function. The DGM/HJB oracle knows the exact Markov state
`(t, X, Y1, Y2)`. The proposed filtered CT-DDPG instead receives a generic
arithmetic bank of 12 observed exponential filters. Signed approximation
readouts remain admissible because the positive part is applied after memory
aggregation. The filtered actor includes a zero-initialized three-coefficient
rate-profile residual (filter level, decay-rate slope, and curvature) learned
only through the CT-DDPG policy gradient; it is not fitted to the true kernel.

## Maintained comparison

CT-DDPG, SAC, and DDPG are reported under two observation protocols:

- `current`: `(t, X)` only.
- `filters`: `(t, X, Z1, ..., Z12)` using normalized observed filters. SAC
  additionally receives 11 adjacent decay-rate differences computed from the
  same bank; these approximate age-weighted memory without oracle information.

History-window variants and PPO are not loaded, evaluated, or plotted by the
maintained notebook. Older history artifacts are preserved under
`archived_history_observation` and ignored by the active suite.

The uncapped subcritical calibration uses `mu0=2.00`, excitation `1.05`, jump
amplitude `clip(0.085 + 0.005 X - 0.010 a, 1e-4, 0.20)`, and cost coefficients
`(c_x, c_a, c_N, c_T)=(0.3000, 0.2250, 0, 0.2250)`. Its worst-case
finite-horizon offspring mass is `0.89343 < 1`. On the final 400-path
common-seed evaluation, costs are approximately:

```text
DGM/HJB oracle    0.1037
CT-DDPG filtered  0.1056
SAC filtered      0.1065
SAC current       0.1093
DDPG filtered     0.1103
CT-DDPG current   0.1219
DDPG current      0.1266
static a=0.26     0.1314
```

The filtered CT policy improves about 13.4% over its current-state ablation and
is numerically best among learned policies. Filtering improves SAC by 2.55%
and DDPG by 12.9% on the notebook's common paths. SAC checkpoint evaluation
resets to the same validation seed block at every comparison, and the selected
full-budget filtered checkpoint has nonzero memory-input weights. The SAC gain
also persists on three independent 800-path blocks. Confidence intervals
condition on one trained policy; paper-level claims should also aggregate
independent training seeds.

All filtered methods start from the same 12 observed readouts. CT-DDPG trains
for 35,000 environment transitions, compared with 220,000 for filtered SAC and
160,000 for filtered DDPG. The notebook now plots final common-test cost against
these complete budgets. This is the defensible sample-efficiency comparison;
the already oracle-adjacent CT-DDPG/SAC terminal-cost gap should not be enlarged
by reducing SAC's information or training budget.

## Entry points

The maintained notebook is:

```text
complex_erlang_experiment/erlang_kernel_comparison.ipynb
```

Cached checkpoints load by default. The tuning cell contains separate force
switches for CT-DDPG, DGM, and SB3. Standalone training commands are:

```powershell
python train_tuned_ct_suite.py --out-dir results --force --max-workers 2
python train_tuned_sb3_suite.py --out-dir results --force --max-workers 4
```

Important outputs are `results/comparison.json`, `figures/benchmark_returns.png`,
`results/sac_filter_fix_validation.json`,
`figures/current_vs_filtered_costs_expanded.png`, the two CT convergence PNGs,
`figures/filtered_cost_vs_training_budget.png`,
the DGM/SB3 convergence PNGs, the three value surfaces,
`figures/policy_path_ensemble_ci.png`, and
`figures/ct_kernel_approximation.png`.
