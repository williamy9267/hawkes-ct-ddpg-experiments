# Section 1 Current-Readout Markovianization

This folder is a separate implementation of the simpler Section 1 method in the
draft.

The model uses a one-dimensional single-exponential controlled Hawkes
jump-diffusion:

```text
dX_t = (b_0 - kappa X_t - b_a a_t) dt
       + [sigma_0 + sigma_x X_t + sigma_a a_t]_+ dW_t
       + [gamma_0 + gamma_x X_t + gamma_a a_t]_+ dN_t

dZ^k_t = -rho_k Z^k_t dt + dN_t

lambda_t = mu(t, X_t, a_t) + alpha q(a_t) Z^{exact}_t
q(a) = 1 - c_eff a / (a_half + a)
```

The true model is classical additive Hawkes. There is no global softplus link.
For finite approximation readouts, signed dictionary coefficients are allowed;
the positive part belongs outside the approximated memory aggregation.

CT-DDPG receives only the online Markov state `(t, X_t, Z^1_t, ..., Z^K_t)`.
It does not use the known intensity formula or the unknown current-readout
coefficient during learning. The DGM/HJB oracle uses the known parameters and
solves the HJB in the exact single-exponential state `(t, X, Z^{exact})`.

The CT-DDPG filter bank uses an arithmetic decay grid
`(beta, 2 beta, ..., K beta)` with `beta=0.325`; the fourth filter is exactly
the true decay `1.30`.

The maintained calibration has no event-count cost and no cap on total
intensity. Its worst-case finite-horizon offspring mass is `0.96009 < 1`.

Open the maintained benchmark notebook from this experiment directory:

```text
single_exponential_check_experiment/current_readout_markov_ct_ddpg.ipynb
```

Generated figures are written to
`single_exponential_check_experiment/figures/`;
numerical results are written to
`single_exponential_check_experiment/results/`.

The notebook plots the CT-DDPG Monte Carlo cost-to-go surface and the DGM/HJB value surface. It also saves the raw CT-DDPG critic surface as a diagnostic.
