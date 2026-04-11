# Run Commands

## Environment Setup

```bash
eval "$($HOME/miniforge3/bin/conda shell.bash hook)" && conda activate gpgomea
```

## PMLB Inference

```bash
# GP-GOMEA（默认算法，time=600s, generations/evaluations 不限）
python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52

# Standard GP（generations=500, time/evaluations 不限）
python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm standard

# SBP-GP（generations=500, time/evaluations 不限）
python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm sbp
```

覆盖终止条件（不传则使用项目默认配置）：

```bash
python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --time 10
python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm standard --generations 100
```
