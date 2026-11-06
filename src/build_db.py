"""Case sampling and evidence database construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import argparse
import math
import time
import data
import database
import evidence
import settings

N_QUINTILES = 5
RNG_SEED = 42


# Evaluation cases
def eligible_months(lib: evidence.TrajectoryLibrary, test_months: list[str]) -> pd.DataFrame:
    """Eligibility audit for all test months."""
    rows = []
    for t in test_months:
        q_valid = lib.has_valid_Q(t)
        n_cand = int(len(lib.eligible(t))) if q_valid else 0
        n_surv = lib.n_survivors(t) if q_valid and n_cand else 0
        n_sel = min(n_surv, evidence.K)
        rows.append(
            dict(
                month=t,
                q_valid=bool(q_valid),
                n_candidates=n_cand,
                n_nms_survivors=int(n_surv),
                n_selected=int(n_sel),
                eligible=bool(q_valid and n_surv >= evidence.K and (n_sel == evidence.K)),
            )
        )
    return pd.DataFrame(rows)


def sample_cases(
    pred_df: pd.DataFrame, months: list[str], rng_seed: int = RNG_SEED
) -> pd.DataFrame:
    """One sampled stock per prediction quintile and eligible month."""
    rng = np.random.default_rng(rng_seed)
    out = []
    for t in sorted(months):
        sub = pred_df[pred_df['month'] == t]
        if sub.empty:
            raise AssertionError(f'{t}: 예측이 없음')
        sub = sub.sort_values(['pred', 'PERMNO'], kind='mergesort').reset_index(drop=True)
        n = len(sub)
        if n < N_QUINTILES:
            raise AssertionError(f'{t}: 종목 {n}개로 5분위 구성 불가')
        quintile_indices = np.floor(np.arange(n) * N_QUINTILES / n).astype(np.int64)
        sub['quintile'] = quintile_indices + 1
        sub['pred_rank_in_month'] = np.arange(1, n + 1)
        sub['pred_pct_in_month'] = np.floor(100.0 * np.arange(n) / n).astype(np.int64)
        sub['n_stocks_in_month'] = n
        for q in range(1, N_QUINTILES + 1):
            pool = sub[sub['quintile'] == q]
            if pool.empty:
                raise AssertionError(f'{t} 분위 {q} 가 비어 있음')
            pick = int(rng.integers(0, len(pool)))
            row = pool.iloc[pick].to_dict()
            row['case_id'] = f"{t}_Q{q}_{int(row['PERMNO'])}"
            out.append(row)
    df = pd.DataFrame(out)
    cols = [
        'case_id',
        'month',
        'PERMNO',
        'quintile',
        'pred',
        'y',
        'pred_rank_in_month',
        'pred_pct_in_month',
        'n_stocks_in_month',
    ]
    df = df[cols]
    df['PERMNO'] = df['PERMNO'].astype(np.int64)
    if df['case_id'].duplicated().any():
        raise AssertionError('case_id 중복 있음')
    if len(df) != len(months) * N_QUINTILES:
        raise AssertionError(f'사례 수 {len(df)} ≠ {len(months)}월 × {N_QUINTILES}')
    return df.reset_index(drop=True)


TOP_LOCAL = 3
TOP_GLOBAL = 3


def top_k_names(values: np.ndarray, k: int) -> list[str]:
    order = np.argsort(-values, kind='stable')
    return [data.RETRIEVAL_ORDER[j] for j in order[:k]]


def select_display_features(phi_row: np.ndarray, P_row: np.ndarray) -> dict:
    """Top-local/global union from two (30,) arrays in RETRIEVAL_ORDER."""
    if phi_row.shape != (data.NF,) or P_row.shape != (data.NF,):
        raise AssertionError(f'입력 shape 오류: phi{phi_row.shape} P{P_row.shape}')
    top_local = top_k_names(np.abs(phi_row), TOP_LOCAL)
    top_global = top_k_names(P_row, TOP_GLOBAL)
    seen: list[str] = []
    for f in top_local + top_global:
        if f not in seen:
            seen.append(f)
    overlap = sorted(set(top_local) & set(top_global), key=data.RETRIEVAL_ORDER.index)
    return {
        'display_features': seen,
        'top_local_features': top_local,
        'top_global_features': top_global,
        'overlap_features': overlap,
        'n_display': len(seen),
    }


def validate_display_features(d: dict) -> None:
    n = d['n_display']
    if not 3 <= n <= 6:
        raise AssertionError(f'|D| = {n} 이 [3,6] 범위 밖임')
    if len(set(d['display_features'])) != n:
        raise AssertionError('D 에 중복이 있음')
    if n != len(set(d['top_local_features']) | set(d['top_global_features'])):
        raise AssertionError('D 가 두 상위집합의 합집합이 아님')
    if len(d['top_local_features']) != TOP_LOCAL or len(d['top_global_features']) != TOP_GLOBAL:
        raise AssertionError('상위 3 개수가 맞지 않음')


RELATION_COLUMNS = [
    'case_id',
    'month',
    'evidence_id',
    'overtaker',
    'overtaken',
    'start_P_overtaker',
    'start_P_overtaken',
    'end_P_overtaker',
    'end_P_overtaken',
]


# Historical context
def _load_monthly_ic(seed: int) -> dict[str, float]:
    """Validated monthly RankIC lookup for the fixed predictor."""
    df = pd.read_parquet(settings.input_path('monthly_ic', seed), columns=['month', 'IC', 'split'])
    if df.empty:
        raise AssertionError('monthly_ic가 비어 있음')
    if df['month'].duplicated().any():
        dup = df.loc[df['month'].duplicated(), 'month'].astype(str).tolist()[:5]
        raise AssertionError(f'monthly_ic month 중복: {dup}')
    vals = pd.to_numeric(df['IC'], errors='coerce')
    if vals.isna().any():
        bad = df.loc[vals.isna(), 'month'].astype(str).tolist()[:5]
        raise AssertionError(f'monthly_ic에 비수치/결측 IC가 있음: {bad}')
    if (~np.isfinite(vals.to_numpy(np.float64))).any():
        raise AssertionError('monthly_ic에 비유한 IC가 있음')
    if ((vals < -1.0) | (vals > 1.0)).any():
        bad = df.loc[(vals < -1.0) | (vals > 1.0), ['month', 'IC']].head().to_dict('records')
        raise AssertionError(f'RankIC 범위 [-1,1] 위반: {bad}')
    return {str(m): float(ic) for m, ic in zip(df['month'], vals)}


def _enrich_analogs(analogs: list[evidence.Analog], ic_lookup: dict[str, float]) -> list[dict]:
    """Historical RankIC and subsequent universe-return evidence."""
    out: list[dict] = []
    for a in analogs:
        s = a.endpoint_s
        if s not in ic_lookup:
            raise AssertionError(f'analog endpoint {s}의 monthly RankIC가 없음')
        ic = float(ic_lookup[s])
        if not math.isfinite(ic) or not -1.0 <= ic <= 1.0:
            raise AssertionError(f'analog endpoint {s}의 RankIC 범위 오류: {ic}')
        out.append(
            {
                'analog_rank': int(a.analog_rank),
                'analog_start': a.analog_start,
                'analog_end': a.analog_end,
                'endpoint_s': s,
                'euclidean_distance': float(a.euclidean_distance),
                'subsequent_return_month': a.subsequent_return_month,
                'subsequent_universe_return': float(a.subsequent_market_return),
                'monthly_rank_ic': ic,
            }
        )
    return out


def _analog_summary(analogs: list[dict]) -> dict:
    """Precomputed return and RankIC summaries across historical analogs."""
    r = np.asarray([a['subsequent_universe_return'] for a in analogs], dtype=np.float64)
    ic = np.asarray([a['monthly_rank_ic'] for a in analogs], dtype=np.float64)
    if r.size != evidence.K or ic.size != evidence.K:
        raise AssertionError(f'analog summary 입력 개수 {r.size}/{ic.size} != K={evidence.K}')
    if not np.isfinite(r).all() or not np.isfinite(ic).all():
        raise AssertionError('analog summary 입력에 비유한 값이 있음')
    return {
        'positive_universe_return_count': int((r > 0).sum()),
        'negative_universe_return_count': int((r < 0).sum()),
        'zero_universe_return_count': int((r == 0).sum()),
        'mean_subsequent_universe_return': float(r.mean()),
        'median_subsequent_universe_return': float(np.median(r)),
        'positive_rank_ic_count': int((ic > 0).sum()),
        'negative_rank_ic_count': int((ic < 0).sum()),
        'zero_rank_ic_count': int((ic == 0).sum()),
        'mean_rank_ic': float(ic.mean()),
        'median_rank_ic': float(np.median(ic)),
        'n_analogs': int(r.size),
    }


# Database build
def main(seed: int) -> dict:
    t0 = time.time()
    data.check_orders()
    inv = data.assert_permutation_invariance()
    print(f'[features] 30피처 계약 OK · 치환 불변성 최대오차 {inv:.2e}', flush=True)
    g = data.load_global_shap(seed)
    print(
        f'[global] Σ_j P = 1 확인 · 월 {len(g.months)}개 (test {len(g.test_months())})', flush=True
    )
    mr = evidence.build_market_return(seed)
    universe_return_lookup = evidence.as_lookup(mr)
    ic_lookup = _load_monthly_ic(seed)
    lib = evidence.TrajectoryLibrary(g, universe_return_lookup)
    print(
        f'[library] endpoint {len(lib.library_endpoints)}개 '
        f'({lib.library_endpoints[0]} .. {lib.library_endpoints[-1]})',
        flush=True,
    )
    missing_ic = [s for s in lib.library_endpoints if s not in ic_lookup]
    if missing_ic:
        raise AssertionError(
            f'library endpoint 중 monthly RankIC 결측 {len(missing_ic)}개: {missing_ic[:5]}'
        )
    audit = eligible_months(lib, g.test_months())
    elig = audit[audit['eligible']].copy()
    if elig.empty:
        raise AssertionError('적격월이 하나도 없음')
    months = elig['month'].tolist()
    print(
        f'[months] 적격월 {len(months)}개  {months[0]} .. {months[-1]}  '
        f'(정확히 5개 생존 월 {int((audit.n_nms_survivors == 5).sum())}개)',
        flush=True,
    )
    pred = data.load_local_shap(seed, months=months, columns=[])
    case_df = sample_cases(pred[['PERMNO', 'month', 'pred', 'y']], months)
    print(f'[cases] {len(case_df)}개 (월 {len(months)} × 분위 5)', flush=True)
    vals = data.extract_model_inputs(case_df)
    val_idx = {c: i for i, c in enumerate(vals['case_id'])}
    loc = data.load_local_shap(seed, months=months, columns=data.PHI_COLS_RETRIEVAL)
    loc_idx = {(int(p), m): i for i, (p, m) in enumerate(zip(loc['PERMNO'], loc['month']))}
    PHI = loc[data.PHI_COLS_RETRIEVAL].to_numpy(np.float64)
    analog_rows: list[dict] = []
    summary_rows: list[dict] = []
    month_analogs: dict[str, list[dict]] = {}
    month_stats: dict[str, dict] = {}
    for t in months:
        raw_analogs = lib.retrieve(t)
        evidence.check_analog_constraints(t, raw_analogs)
        enriched = _enrich_analogs(raw_analogs, ic_lookup)
        stats = _analog_summary(enriched)
        month_analogs[t] = enriched
        month_stats[t] = stats
        for a in enriched:
            analog_rows.append({'month': t, **a})
        summary_rows.append({'month': t, **stats})
    print(
        f'[retrieval] 월당 K={evidence.K} 유사국면 · universe outcome + historical RankIC · '
        f'총 {len(analog_rows)}행',
        flush=True,
    )
    case_records, evidence_records, all_features = ([], [], [])
    relation_records, case_analogs, summaries, registry = ([], [], [], [])
    order = {feature: index for index, feature in enumerate(data.RETRIEVAL_ORDER)}
    for _, case in case_df.iterrows():
        month, permno, case_id = (case['month'], int(case['PERMNO']), case['case_id'])
        phi = PHI[loc_idx[permno, month]]
        importance = g.P[g.index_of(month)]
        display = select_display_features(phi, importance)
        validate_display_features(display)
        selected = display['display_features']
        temporal = evidence.compute(g, month, selected)
        window = lib.window_months(month)
        value_row = vals.iloc[val_idx[case_id]]

        def register(suffix: str, evidence_type: str, layer: str) -> str:
            evidence_id = f'{case_id}#{suffix}'
            registry.append(
                {
                    'case_id': case_id,
                    'evidence_id': evidence_id,
                    'type': evidence_type,
                    'layer': layer,
                    'visible_in': list(settings.VISIBILITY[layer]),
                }
            )
            return evidence_id

        case_records.append(
            {
                'case_id': case_id,
                'month': month,
                'PERMNO': permno,
                'pred': float(case['pred']),
                'quintile': int(case['quintile']),
                'pred_pct_in_month': int(case['pred_pct_in_month']),
                'n_stocks_in_month': int(case['n_stocks_in_month']),
                'n_display': len(selected),
                'traj_window_start': window[0],
                'traj_window_end': window[-1],
            }
        )
        register('PRED', 'PREDICTION', 'base')
        for feature in data.RETRIEVAL_ORDER:
            index = order[feature]
            all_features.append(
                {
                    'case_id': case_id,
                    'PERMNO': permno,
                    'month': month,
                    'feature': feature,
                    'retrieval_order': index,
                    'val_raw': float(value_row[f'raw_{feature}']),
                    'val_z': float(value_row[f'z_{feature}']),
                    'val_pct': int(value_row[f'pct_{feature}']),
                    'phi': float(phi[index]),
                    'phi_abs': abs(float(phi[index])),
                    'global_P': float(importance[index]),
                }
            )
        for position, feature in enumerate(selected):
            value = float(phi[order[feature]])
            te = temporal[feature]
            evidence_records.append(
                {
                    'case_id': case_id,
                    'month': month,
                    'PERMNO': permno,
                    'feature': feature,
                    'retrieval_order': order[feature],
                    'd_position': position,
                    'in_display': True,
                    'phi': value,
                    'phi_abs': abs(value),
                    'phi_direction': (
                        'positive' if value > 0 else 'negative' if value < 0 else 'zero'
                    ),
                    'val_raw': float(value_row[f'raw_{feature}']),
                    'val_z': float(value_row[f'z_{feature}']),
                    'val_pct': int(value_row[f'pct_{feature}']),
                    'global_P': float(importance[order[feature]]),
                    'seq_P': list(te.raw_sequence),
                    'seq_months': list(te.sequence_months),
                    'delta6P': te.window_change_delta6P,
                    'slope': te.trend_slope,
                    'trend': te.trend_direction,
                    'eid_loc': register(f'LOC:{feature}', 'LOCAL_FEATURE', 'base'),
                    'eid_glb': register(f'GLB:{feature}', 'GLOBAL_FEATURE', 'base'),
                    'eid_seq': register(f'SEQ:{feature}', 'TEMPORAL_SEQUENCE', 'raw'),
                    'eid_tmp': register(f'TMP:{feature}', 'TEMPORAL_NUMERICAL', 'numerical'),
                }
            )
        for overtaker in selected:
            first = temporal[overtaker].raw_sequence
            for overtaken in selected:
                second = temporal[overtaken].raw_sequence
                if first[0] < second[0] and first[-1] > second[-1]:
                    relation_records.append(
                        {
                            'case_id': case_id,
                            'month': month,
                            'evidence_id': register(
                                f'OVERTAKES:{overtaker}>{overtaken}', 'OVERTAKES', 'relational'
                            ),
                            'overtaker': overtaker,
                            'overtaken': overtaken,
                            'start_P_overtaker': first[0],
                            'start_P_overtaken': second[0],
                            'end_P_overtaker': first[-1],
                            'end_P_overtaken': second[-1],
                        }
                    )
        register('TRAJ', 'CURRENT_TRAJECTORY', 'analog')
        for analog in month_analogs[month]:
            rank = int(analog['analog_rank'])
            case_analogs.append(
                {
                    'case_id': case_id,
                    'month': month,
                    'analog_rank': rank,
                    'analog_start': analog['analog_start'],
                    'analog_end': analog['analog_end'],
                    'euclidean_distance': analog['euclidean_distance'],
                    'return_month': analog['subsequent_return_month'],
                    'subsequent_universe_return': analog['subsequent_universe_return'],
                    'monthly_rank_ic': analog['monthly_rank_ic'],
                    'eid_analog': register(f'ANALOG:{rank}', 'HISTORICAL_ANALOG', 'analog'),
                    'eid_universe': register(
                        f'UNIVERSE_OUTCOME:{rank}', 'ANALOG_UNIVERSE_OUTCOME', 'analog'
                    ),
                    'eid_perf': register(
                        f'MODEL_PERF:{rank}', 'ANALOG_MODEL_PERFORMANCE', 'analog'
                    ),
                }
            )
            register(f'SIMILAR:{rank}', 'SIMILAR_TO', 'analog')
            register(f'FOLLOWED:{rank}', 'FOLLOWED_BY', 'analog')
            register(f'EVALUATED:{rank}', 'EVALUATED_BY', 'analog')
        summaries.append(
            {
                'case_id': case_id,
                'month': month,
                'evidence_id': register('ANALOGSUM', 'ANALOG_SUMMARY', 'analog'),
                **month_stats[month],
            }
        )
    tables = {
        'eligible_months': elig.reset_index(drop=True),
        'analogs_by_month': pd.DataFrame(analog_rows),
        'cases': pd.DataFrame(case_records),
        'feature_metadata': database.feature_metadata_table(),
        'evidence_registry': pd.DataFrame(registry),
        'relations': pd.DataFrame(relation_records, columns=RELATION_COLUMNS),
        'analogs': pd.DataFrame(case_analogs),
        'analog_summary': pd.DataFrame(summaries),
        'features_all30': pd.DataFrame(all_features),
        'evidence': pd.DataFrame(evidence_records),
    }
    counts = database.write_database(tables, settings.database_path(seed))
    summary = {
        'seed': seed,
        'database': str(settings.database_path(seed)),
        'table_rows': counts,
        'n_cases': len(case_records),
        'n_eligible_months': len(months),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    settings.write_json(settings.MANIFESTS / f'build_evidence_seed{seed}.json', summary)
    print(summary, flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args().seed)
