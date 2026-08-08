Use `~/venv/torch` for Python work in this repository.

Example:

```bash
source ~/venv/torch/bin/activate
```

If `hostname` is `ne1-login`, compute-heavy jobs must be submitted through Slurm.
Otherwise, assume this is a GPU compute node and test things directly here when
reasonable, but still submit long-running jobs through Slurm.

Slurm GPU requests can target specific GPU types. For L40S, use:

```bash
sbatch -p gpu --gres gpu:l40s:1 ...
```

For RTX PRO 6000 Blackwell nodes, use:

```bash
sbatch -p gpu --gres gpu:b6k:1 ...
```
