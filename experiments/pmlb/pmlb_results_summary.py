import argparse
import os
import re

import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize PMLB batch inference results by dataset family')
    parser.add_argument('--input_csv', type=str, nargs='+', required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    args = parser.parse_args()

    summary_rows = []
    for input_csv in args.input_csv:
        df = pd.read_csv(input_csv)
        dataset_names = df['dataset'].astype(str)
        algorithm = df['algorithm'].astype(str).iloc[0]
        noise_match = re.search(r'_noise([^_]+)\.csv$', os.path.basename(input_csv))
        noise = noise_match.group(1)

        for group_name, group_mask in [
            ('Feynman', dataset_names.str.startswith('feynman_')),
            ('Strogatz', dataset_names.str.startswith('strogatz_')),
            ('Black-box', ~dataset_names.str.startswith('feynman_') & ~dataset_names.str.startswith('strogatz_')),
        ]:
            group_df = df.loc[group_mask].copy()

            r2_raw = pd.to_numeric(group_df['r2'], errors='coerce')
            r2_finite_mask = np.isfinite(r2_raw)
            r2_clean = r2_raw.where(r2_finite_mask, 0.0).clip(lower=0)

            ok_mask = group_df['status'].eq('ok')
            complexity_raw = pd.to_numeric(group_df.loc[ok_mask, 'complexity'], errors='coerce')
            seconds_raw = pd.to_numeric(group_df.loc[ok_mask, 'seconds'], errors='coerce')
            complexity_values = complexity_raw[np.isfinite(complexity_raw)]
            seconds_values = seconds_raw[np.isfinite(seconds_raw)]

            summary_rows.append({
                'algorithm': algorithm,
                'noise': noise,
                'group': group_name,
                'r2_mean': r2_clean.mean(),
                'r2_std': r2_clean.std(ddof=0),
                'r2_valid_count': int((r2_finite_mask & (r2_raw >= 0)).sum()),
                'total_count': int(len(group_df)),
                'recovery_rate': (r2_clean > 0.9).mean(),
                'complexity_mean': complexity_values.mean(),
                'complexity_std': complexity_values.std(ddof=0),
                'complexity_count': int(len(complexity_values)),
                'seconds_mean': seconds_values.mean(),
                'seconds_std': seconds_values.std(ddof=0),
                'seconds_count': int(len(seconds_values)),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df[
        [
            'algorithm',
            'noise',
            'group',
            'r2_mean',
            'r2_std',
            'r2_valid_count',
            'total_count',
            'recovery_rate',
            'complexity_mean',
            'complexity_std',
            'complexity_count',
            'seconds_mean',
            'seconds_std',
            'seconds_count',
        ]
    ]

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    summary_df.to_csv(args.output_csv, index=False)

    print(f'Summary saved to {args.output_csv}')
    print(summary_df.to_string(index=False, float_format=lambda value: f'{value:.6f}'))
