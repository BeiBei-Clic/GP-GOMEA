import argparse
import os
import sys
import time
import traceback

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pythonpkg'))
from pyGPGOMEA import GPGOMEARegressor

from pmlb_inference import load_dataset, simplify_expression, ALGORITHM_DEFAULTS

CSV_COLUMNS = [
    'algorithm', 'dataset', 'status', 'n_features', 'n_rows',
    'r2', 'rmse', 'complexity', 'seconds', 'error', 'expression',
]


def discover_regression_datasets(datasets_dir):
    """扫描 datasets_dir 下所有子目录，通过 metadata.yaml 过滤回归数据集。"""
    datasets = []
    for name in sorted(os.listdir(datasets_dir)):
        meta_path = os.path.join(datasets_dir, name, 'metadata.yaml')
        if not os.path.isfile(meta_path):
            continue
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        if meta.get('task') == 'regression':
            datasets.append(name)
    return datasets


def run_single(dataset, n_rows, seed, algorithm, overrides, noise_strength=0.0, noise_seed=0):
    """对单个数据集跑推理，返回结果 dict。"""
    X, y = load_dataset(dataset, n_rows)
    n_features = X.shape[1]
    actual_rows = X.shape[0]

    if noise_strength > 0:
        rng = np.random.RandomState(noise_seed)
        y = y * (1.0 + noise_strength * rng.randn(len(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    kwargs = {**ALGORITHM_DEFAULTS[algorithm], **overrides, 'seed': seed}
    ea = GPGOMEARegressor(**kwargs)

    t0 = time.time()
    ea.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred = ea.predict(X_test)
    r2 = round(r2_score(y_test, y_pred), 6)
    rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 6)
    complexity = ea.get_n_nodes()

    raw_model = ea.get_model()
    expression = simplify_expression(raw_model)

    return {
        'algorithm': algorithm,
        'dataset': dataset,
        'status': 'ok',
        'n_features': n_features,
        'n_rows': actual_rows,
        'r2': r2,
        'rmse': rmse,
        'complexity': complexity,
        'seconds': elapsed,
        'error': '',
        'expression': expression,
    }


def main():
    parser = argparse.ArgumentParser(description='PMLB batch inference with GP-GOMEA')
    parser.add_argument('--datasets_dir', type=str,
                        default='/home/xyh/Symbolic_Regression/pmlb/datasets')
    parser.add_argument('--algorithm', type=str, default='gomea',
                        choices=['gomea', 'standard', 'sbp'])
    parser.add_argument('--max_rows', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_limit', type=int, default=None,
                        help='Limit number of datasets (for smoke test)')
    parser.add_argument('--time', type=int, default=None)
    parser.add_argument('--generations', type=int, default=None)
    parser.add_argument('--evaluations', type=int, default=None)
    parser.add_argument('--noise_strength', type=float, default=0.0,
                        help='Multiplicative noise strength: y *= (1 + strength * N(0,1))')
    parser.add_argument('--noise_seed', type=int, default=0,
                        help='Random seed for noise generation')
    args = parser.parse_args()

    noise_tag = f'_noise{args.noise_strength}'
    output_csv = os.path.join(os.path.dirname(__file__), 'results', f'pmlb_results_{args.algorithm}{noise_tag}.csv')

    overrides = {
        k: v for k, v in [
            ('time', args.time),
            ('generations', args.generations),
            ('evaluations', args.evaluations),
        ] if v is not None
    }

    datasets = discover_regression_datasets(args.datasets_dir)
    print(f'Found {len(datasets)} regression datasets')

    if args.dataset_limit:
        datasets = datasets[:args.dataset_limit]
        print(f'Limited to first {args.dataset_limit} datasets')

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    write_header = not os.path.exists(output_csv)

    if not write_header:
        existing = pd.read_csv(output_csv)['dataset'].tolist()
        datasets = [d for d in datasets if d not in existing]
        print(f'Skipped {len(existing)} already processed datasets, {len(datasets)} remaining')

    total = len(datasets)
    for i, dataset in enumerate(datasets, 1):
        print(f'\n[{i}/{total}] {dataset}')
        row = None
        error_msg = ''
        try:
            row = run_single(dataset, args.max_rows, args.seed, args.algorithm, overrides, args.noise_strength, args.noise_seed + i)
        except Exception as e:
            error_msg = f'{type(e).__name__}: {e}'
            print(f'  ERROR: {error_msg}')
            traceback.print_exc()
            row = {
                'algorithm': args.algorithm,
                'dataset': dataset,
                'status': 'error',
                'n_features': '',
                'n_rows': '',
                'r2': '',
                'rmse': '',
                'complexity': '',
                'seconds': '',
                'error': error_msg,
                'expression': '',
            }

        row_df = pd.DataFrame([row], columns=CSV_COLUMNS)
        row_df.to_csv(output_csv, mode='a', header=write_header, index=False)
        write_header = False

        with open(output_csv, 'a'):
            os.fsync(os.open(output_csv, os.O_RDONLY))

        if row['status'] == 'ok':
            print(f'  R2={row["r2"]}  RMSE={row["rmse"]}  complexity={row["complexity"]}  {row["seconds"]}s')

    print(f'\nDone. Results saved to {output_csv}')


if __name__ == '__main__':
    main()
