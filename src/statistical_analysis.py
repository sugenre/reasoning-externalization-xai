"""Paired month-cluster bootstrap and Holm adjustment."""

from __future__ import annotations

import argparse
import json
import numpy as np
import pandas as pd
import settings

METRICS = [
    'evidence_faithfulness',
    'diagnostic_temporal_accuracy',
    'relational_detection_accuracy',
    'relational_set_f1',
]
INCOMPARABLE = {'diagnostic_temporal_accuracy': {'C0-Evidence Control'}}


def incomparable_reason(metric: str, cond_a: str, cond_b: str) -> str | None:
    """Incomparable scoring criteria across conditions."""
    bad = INCOMPARABLE.get(metric, set())
    hit = sorted(bad & {cond_a, cond_b})
    if not hit:
        return None
    return f"scoring_criterion_differs_in:{','.join(hit)}"


# Paired bootstrap
def paired_month_cluster_bootstrap(
    df: pd.DataFrame, metric: str, cond_a: str, cond_b: str, n_boot: int, rng: np.random.Generator
) -> dict:
    a = df[df['condition'] == cond_a].set_index('case_id')
    b = df[df['condition'] == cond_b].set_index('case_id')
    common = a.index.intersection(b.index)
    d = pd.DataFrame(
        {
            'month': a.loc[common, 'month'],
            'diff': a.loc[common, metric].to_numpy(float) - b.loc[common, metric].to_numpy(float),
        }
    )
    d = d[np.isfinite(d['diff'])]
    if d.empty:
        return {
            'metric': metric,
            'contrast': f'{cond_a} - {cond_b}',
            'n_cases': 0,
            'n_pairs': 0,
            'n_months': 0,
            'mean_diff': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'p_value': float('nan'),
            'excluded_reason': 'no_finite_paired_cases',
        }
    per_month = d.groupby('month')['diff'].mean()
    months = per_month.index.to_numpy()
    vals = per_month.to_numpy(float)
    M = len(months)
    idx = rng.integers(0, M, size=(n_boot, M))
    boot = vals[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    reason = incomparable_reason(metric, cond_a, cond_b)
    if reason is not None:
        return {
            'metric': metric,
            'contrast': f'{cond_a} - {cond_b}',
            'n_cases': 0,
            'n_pairs': 0,
            'n_months': 0,
            'mean_diff': float('nan'),
            'ci_low': float('nan'),
            'ci_high': float('nan'),
            'p_value': float('nan'),
            'boot_frac_gt_0': float('nan'),
            'excluded_reason': reason,
        }
    return {
        'metric': metric,
        'contrast': f'{cond_a} - {cond_b}',
        'n_cases': int(len(d)),
        'n_pairs': int(len(d)),
        'n_months': M,
        'mean_diff': float(vals.mean()),
        'ci_low': float(lo),
        'ci_high': float(hi),
        'p_value': float(min(1.0, p)),
        'boot_frac_gt_0': float((boot > 0).mean()),
        'excluded_reason': '',
    }


# Multiplicity adjustment
def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjustment, preserving missing values."""
    idx = [i for i, p in enumerate(pvals) if p == p]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    out = [float('nan')] * len(pvals)
    running = 0.0
    for r, i in enumerate(order):
        adj = min(1.0, (m - r) * pvals[i])
        running = max(running, adj)
        out[i] = running
    return out


# Condition contrasts
def main(role: str, seed: int) -> dict:
    settings.E_STATS.mkdir(parents=True, exist_ok=True)
    cfg = settings.frozen_config()['statistics']
    df = pd.read_parquet(settings.E_AUTO / f'case_metrics_{role}_seed{seed}.parquet')
    rng = np.random.default_rng(cfg['rng_seed'])
    primary_rows = []
    for metric in METRICS:
        for a, b in cfg['primary_contrasts']:
            primary_rows.append(
                paired_month_cluster_bootstrap(df, metric, a, b, cfg['n_resamples'], rng)
            )
    pdf = pd.DataFrame(primary_rows)
    pdf['p_holm'] = np.nan
    for metric, blk in pdf.groupby('metric'):
        pdf.loc[blk.index, 'p_holm'] = holm(blk['p_value'].tolist())
    pdf.to_csv(settings.E_STATS / f'primary_contrasts_{role}_seed{seed}.csv', index=False)
    conds = sorted(df['condition'].unique())
    expl = []
    for metric in METRICS:
        for i in range(len(conds)):
            for j in range(len(conds)):
                if i <= j:
                    continue
                expl.append(
                    paired_month_cluster_bootstrap(
                        df, metric, conds[i], conds[j], cfg['n_resamples'], rng
                    )
                )
    edf = pd.DataFrame(expl)
    edf.to_csv(settings.E_STATS / f'exploratory_contrasts_{role}_seed{seed}.csv', index=False)
    summary = {
        'role': role,
        'seed': seed,
        'design': {
            'cluster_unit': cfg['cluster_unit'],
            'bootstrap': cfg['bootstrap'],
            'n_resamples': cfg['n_resamples'],
            'rng_seed': cfg['rng_seed'],
            'ci': cfg['ci'],
            'multiplicity': cfg['multiplicity'],
            'primary_contrasts': cfg['primary_contrasts'],
        },
        'primary': pdf.to_dict('records'),
        'note': 'stock-month 사례는 독립이 아님. 월 클러스터로 재표본. 탐색적 대조는 별도 파일로 저장, Holm 보정 대상에서 제외.',
    }
    print(pdf.to_string(index=False), flush=True)
    settings.write_json(settings.MANIFESTS / f'statistics_{role}_seed{seed}.json', summary)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--role',
        default='primary',
        choices=[
            key
            for key, value in settings.llm_config().items()
            if isinstance(value, dict) and 'model_id' in value
        ],
    )
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()
    main(a.role, a.seed)
