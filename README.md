# Final Hawkes CT-DDPG experiments

The three notebooks below are the entry points for the paper's numerical experiments.
Each experiment folder contains its code, figures, final common-seed comparison in
`results/comparison.json`, and saved policy checkpoints.

| Experiment | Notebook | Final test paths |
| --- | --- | ---: |
| Single exponential | [current_readout_markov_ct_ddpg.ipynb](single_exponential_check_experiment/current_readout_markov_ct_ddpg.ipynb) | 300 |
| Erlang-2 | [erlang_kernel_comparison.ipynb](complex_erlang_experiment/erlang_kernel_comparison.ipynb) | 400 |
| Power law | [powerlaw_kernel_comparison.ipynb](powerlaw_kernel_experiment/powerlaw_kernel_comparison.ipynb) | 800 |

The [numerical setup and interpretation](numerical_setup_and_interpretation.tex)
records the parameters, results, and figure references. The
[benchmark notes](BENCHMARK_ALGORITHMS.md) describe the evaluation protocol.
The static actions used in the final comparisons are 0.39, 0.26, and 0.34,
respectively. The comparison JSON files store returns, so reported cost equals
negative return.

## Rerun the notebooks

Use Python 3.12. From the repository root, install the
dependencies in a Python environment and launch JupyterLab:

```text
python -m pip install -r requirements.txt
python -m jupyter lab
```

Open each notebook in its experiment folder and select the kernel for that
environment. The saved SAC and DDPG models were serialized with NumPy 2.0.1 and
Stable-Baselines3 2.8.0. Keep the retraining flags off to use the saved policies.
The single-exponential oracle cache was created on CPU. On a CUDA-enabled
machine, start Jupyter with `CUDA_VISIBLE_DEVICES=-1` to reuse that cache with the
original notebook. In PowerShell, set `$env:CUDA_VISIBLE_DEVICES = "-1"` before
launching JupyterLab.

The notebooks write figures and comparison files when run. Run from an extracted
copy to retain the packaged results for reference.
