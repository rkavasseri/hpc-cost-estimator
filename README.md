# hpc-cost-estimator

A small, dependency-free CLI for estimating GPU/HPC compute costs for ML
research pipelines — grant-budget-ready output, straight from the terminal.
No PyPI package, no server, just a script.

Built for the gap between commercial cloud pricing calculators (built for
procurement teams) and institutional HPC allocations (built around Service
Units and grant contingency lines). See [the writeup](./ARTICLE.md) for the
full reasoning, assumptions, and a couple of mistakes made along the way.

## Quick start

```bash
python3 hpc_cost_estimator.py --params 200M --tokens 50B --gpu h100 --num-gpus 4
```

```bash
python3 hpc_cost_estimator.py --config examples/pipeline_example_vlm.json --compare-commercial
```

No dependencies beyond the Python standard library. Works with Python 3.7+.

## What it does

Costs a multi-phase research pipeline (pretraining, fine-tuning, ablations,
evaluation, ...) three ways per phase, in priority order:

1. **Explicit GPU-hours**, if you already know them.
2. **Wall-clock hours × GPU count**, if you know how long it'll run.
3. **A FLOPs-based estimate** from model size + training tokens/images, at a
   configurable Model FLOPs Utilization (MFU), if you only know the model and
   dataset size.

It then applies named, adjustable grant-budget contingency layers (restarts,
exploratory runs, reviewer-requested experiments) and — optionally — compares
your institutional SU rate against rough commercial cloud tiers.

## Files

```
hpc_cost_estimator.py          # the CLI — stdlib only, no pip install needed
examples/
  pipeline_example_vlm.json    # example: CV/VLM research pipeline
ARTICLE.md                     # full writeup: methodology, assumptions, limitations
```

## CLI reference

| Flag | Purpose |
|---|---|
| `--config FILE` | Multi-phase pipeline JSON |
| `--params`, `--tokens`, `--images` | Quick single-phase mode |
| `--gpu {h100,a100}` | GPU type (quick mode) |
| `--mfu FLOAT` | Override default Model FLOPs Utilization (default 0.30) |
| `--su-rate FLOAT` | Override institutional SU rate |
| `--compare-commercial` | Add specialized-cloud and hyperscaler comparison |
| `--csv FILE` | Write results to CSV |
| `--no-contingency` | Skip the contingency section |

## Calibration

The FLOPs estimator (`6 × params × tokens`) is checked against a published
reference: a 1.3B-parameter model on 100B tokens costs roughly 1,000 H100
GPU-hours in practice. At MFU = 22%, this tool reproduces that almost exactly
(995.8 hours). Default MFU is 30% — override it per-phase once you have your
own team's real throughput to calibrate against. See `ARTICLE.md` for the full
discussion of where this estimator is (and isn't) trustworthy.

## License

MIT — see [LICENSE](./LICENSE). Use it, fork it, adapt the rates to your own
cluster.

## Contributing

This was built for one lab's workflow and generalized just enough to be
useful elsewhere. If your institution's SU structure, contingency norms, or
GPU lineup don't fit the defaults, PRs adjusting `GPU_DEFAULTS`,
`COMMERCIAL_RATES`, or `DEFAULT_CONTINGENCY` in `hpc_cost_estimator.py` are
welcome.
