# Run Commands

## Environment Setup

```bash
# uv（推荐）
uv sync
uv run python <script>

# conda（旧方式）
eval "$($HOME/miniforge3/bin/conda shell.bash hook)" && conda activate gpgomea
```

## PMLB Inference

```bash
# GP-GOMEA（默认算法，time=2000s, generations/evaluations 不限）
uv run python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52

# Standard GP（time=2000s, generations=500）
uv run python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm standard

# SBP-GP（time=2000s, generations=500）
uv run python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm sbp
```

覆盖终止条件（不传则使用项目默认配置）：

```bash
uv run python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --time 10
uv run python experiments/pmlb/pmlb_inference.py --dataset 192_vineyard --n_rows 52 --algorithm standard --generations 100
```

## PMLB Batch Inference

结果按算法分文件保存至 `experiments/pmlb/results/pmlb_results_{algorithm}.csv`（有噪声时为 `pmlb_results_{algorithm}_noise{strength}.csv`）。

```bash
# Smoke test（3 个数据集，每个 5 秒）
uv run python experiments/pmlb/pmlb_batch_inference.py --algorithm gomea --dataset_limit 3 --time 5 --max_rows 50

# 全量推理（271 个回归数据集，三种算法各跑一次，默认 time=2000s）
uv run python experiments/pmlb/pmlb_batch_inference.py --algorithm gomea --noise_strength 0.1
uv run python experiments/pmlb/pmlb_batch_inference.py --algorithm standard --noise_strength 0.1
uv run python experiments/pmlb/pmlb_batch_inference.py --algorithm sbp --noise_strength 0.1

# 加乘性噪声（y *= (1 + noise_strength * N(0,1))）
uv run python experiments/pmlb/pmlb_batch_inference.py --algorithm gomea --noise_strength 0.1
```
