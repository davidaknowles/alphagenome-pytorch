Use `~/venv/torch` for Python work in this repository.

Example:

```bash
source ~/venv/torch/bin/activate
```

Slurm GPU requests can target specific GPU types. For L40S, use:

```bash
sbatch -p gpu --gres gpu:l40s:1 ...
```

For RTX PRO 6000 Blackwell nodes, use:

```bash
sbatch -p gpu --gres gpu:b6k:1 ...
```
