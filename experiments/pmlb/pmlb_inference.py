import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import sympy
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'pythonpkg'))
from pyGPGOMEA import GPGOMEARegressor

PMLB_DIR = '/home/xyh/Symbolic_Regression/pmlb/datasets'
RESULTS_CSV = os.path.join(os.path.dirname(__file__), 'results', 'pmlb_results.csv')


def load_dataset(name, n_rows):
    path = os.path.join(PMLB_DIR, name, f'{name}.tsv.gz')
    df = pd.read_csv(path, sep='\t')
    df = df.head(n_rows)
    X = df.drop(columns=['target']).values
    y = df['target'].values
    return X, y


def simplify_expression(expr_str):
    expr_str = expr_str.replace('p/', '/').replace('plog', 'log')
    simplified = str(sympy.simplify(expr_str))
    if simplified == '0':
        return expr_str
    return simplified


# Project defaults from params/params_gomea.txt, params_sgp.txt, params_semback.txt
ALGORITHM_DEFAULTS = {
    'gomea': dict(
        gomea=True, gomfos='LT',
        ims='5_1', initmaxtreeheight=4, syntuniqinit=1000,
        time=600, generations=-1, evaluations=-1,
    ),
    'standard': dict(
        gomea=False,
        initmaxtreeheight=6, maxtreeheight=12, popsize=1000, syntuniqinit=1000,
        subcross=0.5, submut=0.5, reproduction=0.0,
        tournament=4, elitism=1,
        time=-1, generations=500, evaluations=-1,
    ),
    'sbp': dict(
        gomea=False,
        initmaxtreeheight=6, maxtreeheight=12, popsize=1000, syntuniqinit=1000,
        subcross=0.0, submut=0.0,
        sblibtype='p_12_9999_l_n', sbrdo=1.0,
        tournament=4, elitism=1,
        time=-1, generations=500, evaluations=-1,
    ),
}


def run_inference(dataset, n_rows, seed, algorithm, overrides):
    print(f'Loading dataset: {dataset} (n_rows={n_rows})')
    X, y = load_dataset(dataset, n_rows)
    print(f'  shape: X={X.shape}, y={y.shape}')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    kwargs = {**ALGORITHM_DEFAULTS[algorithm], **overrides, 'seed': seed}
    print(f'Running {algorithm}...')
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

    print(f'  R2: {r2}')
    print(f'  RMSE: {rmse}')
    print(f'  Complexity (nodes): {complexity}')
    print(f'  Time: {elapsed}s')
    print(f'  Expression: {expression}')

    row = {
        'algorithm': algorithm,
        'dataset': dataset,
        'n_rows': n_rows,
        'r2': r2,
        'rmse': rmse,
        'time_seconds': elapsed,
        'complexity': complexity,
        'expression': expression,
    }

    results_df = pd.DataFrame([row])
    write_header = not os.path.exists(RESULTS_CSV)
    results_df.to_csv(RESULTS_CSV, mode='a', header=write_header, index=False)
    print(f'Results appended to {RESULTS_CSV}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PMLB dataset inference with GP-GOMEA')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g. 192_vineyard)')
    parser.add_argument('--n_rows', type=int, default=200, help='Number of rows to use')
    parser.add_argument('--time', type=int, default=None, help='Time limit in seconds (default: gomea=600, standard/sbp=no limit)')
    parser.add_argument('--generations', type=int, default=None, help='Max generations, -1 for no limit (default: gomea=no limit, standard/sbp=500)')
    parser.add_argument('--evaluations', type=int, default=None, help='Max evaluations, -1 for no limit')
    parser.add_argument('--algorithm', type=str, default='gomea',
                        choices=['gomea', 'standard', 'sbp'],
                        help='Algorithm: gomea (GP-GOMEA), standard (Standard GP), sbp (SBP-GP)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    overrides = {k: v for k, v in [('time', args.time), ('generations', args.generations), ('evaluations', args.evaluations)] if v is not None}
    run_inference(args.dataset, args.n_rows, args.seed, args.algorithm, overrides)
