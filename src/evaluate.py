"""Condition-level evaluation of stored model outputs."""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import database
import prompts as prompt_templates
import scoring
import settings


def _nanmean(x) -> float:
    vals = [float(v) for v in x if pd.notna(v)]
    return float(np.mean(vals)) if vals else float('nan')


def _mean_bool(x) -> float:
    vals = [bool(v) for v in x if pd.notna(v)]
    return float(np.mean(vals)) if vals else float('nan')


# Parse failures
def _parse_failure_row(base: dict, n_repair_attempts: int) -> dict:
    """Parse-failure record with undefined content metrics."""
    return {
        **base,
        'parseable': False,
        'n_factual_claims': 0,
        'n_supported': 0,
        'evidence_faithfulness': np.nan,
        'expected_mode': None,
        'diagnostic_temporal_accuracy': np.nan,
        'evidence_control_abstention_correct': np.nan,
        'n_focal_trend_claims': np.nan,
        'n_focal_abstain_claims': np.nan,
        'relational_detection_accuracy': np.nan,
        'relational_precision': np.nan,
        'relational_set_recall': np.nan,
        'relational_set_f1': np.nan,
        'n_relational_pred': np.nan,
        'n_relational_gold': np.nan,
        'n_relational_tp': np.nan,
        'n_relational_fp': np.nan,
        'relational_gold_empty': np.nan,
        'relational_abstained': np.nan,
        'n_repair_attempts': n_repair_attempts,
    }


# Condition evaluation
def main(role: str, seed: int, variant: str = 'B5', generations: Path | None = None) -> dict:
    tag = '' if variant == 'B5' else f'_{variant}'
    generation_path = (
        generations or settings.A_GENERATIONS / f'generations_{role}_seed{seed}{tag}.jsonl'
    )
    with generation_path.open(encoding='utf-8') as handle:
        gens = [json.loads(line) for line in handle if line.strip()]
    settings.E_AUTO.mkdir(parents=True, exist_ok=True)
    if not gens:
        raise AssertionError('generation 결과가 비어 있음')
    verdict_rows: list[dict] = []
    case_rows: list[dict] = []
    cached_cases: dict[str, database.CaseEvidence] = {}
    seen: set[tuple[str, str]] = set()
    with closing(database.connect(seed)) as con:
        database.validate_database(con)
        for r in gens:
            cid = r['case_id']
            if cid not in cached_cases:
                cached_cases[cid] = database.CaseEvidence(con, cid)
            evidence = cached_cases[cid]
            if r['month'] != evidence.case['month']:
                raise ValueError(f'Generation month does not match DB case: {cid}')
            key = (cid, r['condition'])
            if key in seen:
                raise ValueError(f'Duplicate generated case/condition: {key}')
            seen.add(key)
            cond = r['condition']
            if cond not in settings.CONDITIONS:
                raise ValueError(f'알 수 없는 condition: {cond}')
            base = {
                'case_id': cid,
                'month': r['month'],
                'condition': cond,
                'role': role,
                'model_id': r['model_id'],
            }
            if r['parsed'] is None:
                case_rows.append(_parse_failure_row(base, r['n_repair_attempts']))
                continue
            parsed = r['parsed']
            ef = scoring.evidence_faithfulness(parsed, evidence, cond)
            task = scoring.task_metrics(parsed, evidence, cond)
            for v in ef['verdicts']:
                verdict_rows.append({**base, **v, 'cited_ids': '|'.join(v['cited_ids'])})
            case_rows.append(
                {
                    **base,
                    'parseable': True,
                    'n_factual_claims': ef['n_factual_claims'],
                    'n_supported': ef['n_supported'],
                    'evidence_faithfulness': ef['evidence_faithfulness'],
                    'expected_mode': task['expected_mode'],
                    'diagnostic_temporal_accuracy': task['diagnostic_temporal_accuracy'],
                    'evidence_control_abstention_correct': (
                        task['diagnostic_temporal_accuracy']
                        if cond == settings.C0_EVIDENCE_CONTROL
                        else np.nan
                    ),
                    'n_focal_trend_claims': task['n_focal_trend_claims'],
                    'n_focal_abstain_claims': task['n_focal_abstain_claims'],
                    'relational_detection_accuracy': task['relational_detection_accuracy'],
                    'relational_precision': task['relational_precision'],
                    'relational_set_recall': task['relational_set_recall'],
                    'relational_set_f1': task['relational_set_f1'],
                    'n_relational_pred': task['n_relational_pred'],
                    'n_relational_gold': task['n_relational_gold'],
                    'n_relational_tp': task['n_relational_tp'],
                    'n_relational_fp': task['n_relational_fp'],
                    'relational_gold_empty': task['relational_gold_empty'],
                    'relational_abstained': task['relational_abstained'],
                    'n_repair_attempts': r['n_repair_attempts'],
                }
            )
    vdf = pd.DataFrame(verdict_rows)
    cdf = pd.DataFrame(case_rows)
    if cdf.empty:
        raise AssertionError('case-level evaluation 결과가 비어 있음')
    vdf.to_parquet(settings.E_AUTO / f'claim_verdicts_{role}_seed{seed}{tag}.parquet', index=False)
    cdf.to_parquet(settings.E_AUTO / f'case_metrics_{role}_seed{seed}{tag}.parquet', index=False)
    cond_order = {c: i for i, c in enumerate(settings.CONDITIONS)}
    unknown = sorted(set(cdf['condition']) - set(cond_order))
    if unknown:
        raise AssertionError(f'evaluation 결과에 정의되지 않은 condition이 있음: {unknown}')
    rows: list[dict] = []
    for cond in settings.CONDITIONS:
        blk = cdf[cdf['condition'] == cond]
        if blk.empty:
            raise AssertionError(f'{cond}: 평가 case가 하나도 없음')
        parsed_blk = blk[blk['parseable']]
        empty_gold = parsed_blk[parsed_blk['relational_gold_empty'] == True]
        nonempty_gold = parsed_blk[parsed_blk['relational_gold_empty'] == False]
        rows.append(
            {
                'condition': cond,
                'n_cases': int(len(blk)),
                'n_valid': int(len(parsed_blk)),
                'parse_rate': float(blk['parseable'].mean()),
                'mean_factual_claims': (
                    float(parsed_blk['n_factual_claims'].mean())
                    if len(parsed_blk)
                    else float('nan')
                ),
                'mean_supported_claims': (
                    float(parsed_blk['n_supported'].mean()) if len(parsed_blk) else float('nan')
                ),
                'evidence_faithfulness': _nanmean(parsed_blk['evidence_faithfulness']),
                'diagnostic_temporal_accuracy': _nanmean(
                    parsed_blk['diagnostic_temporal_accuracy']
                ),
                'evidence_control_abstention_accuracy': _nanmean(
                    parsed_blk['evidence_control_abstention_correct']
                ),
                'relational_detection_accuracy': _nanmean(
                    parsed_blk['relational_detection_accuracy']
                ),
                'relational_precision': _nanmean(parsed_blk['relational_precision']),
                'relational_set_recall': _nanmean(parsed_blk['relational_set_recall']),
                'relational_set_f1': _nanmean(parsed_blk['relational_set_f1']),
                'relational_abstention_correct_rate': _mean_bool(
                    empty_gold['relational_abstained']
                ),
                'relational_detection_accuracy_nonempty_gold': _nanmean(
                    nonempty_gold['relational_detection_accuracy']
                ),
                'n_relational_gold_empty': int(len(empty_gold)),
                'n_relational_gold_nonempty': int(len(nonempty_gold)),
                'mean_relational_gold_per_case': _nanmean(parsed_blk['n_relational_gold']),
                'mean_relational_pred_per_case': _nanmean(parsed_blk['n_relational_pred']),
                'mean_repair_attempts': float(blk['n_repair_attempts'].mean()),
            }
        )
    sdf = pd.DataFrame(rows)
    sdf['_order'] = sdf['condition'].map(cond_order)
    sdf = sdf.sort_values('_order').drop(columns=['_order'])
    sdf.to_csv(settings.E_AUTO / f'condition_summary_{role}_seed{seed}{tag}.csv', index=False)
    if len(vdf):
        factual_types = set(prompt_templates.FACTUAL_CLAIM_TYPES)
        factual_bad = vdf[(vdf['supported'] == False) & vdf['claim_type'].isin(factual_types)]
        all_bad = vdf[(vdf['supported'] == False) & (vdf['claim_type'] != 'caution')]
        top_factual_reasons = factual_bad['reason'].value_counts().head(12).to_dict()
        top_all_reasons = all_bad['reason'].value_counts().head(12).to_dict()
    else:
        top_factual_reasons = {}
        top_all_reasons = {}
    summary = {
        'role': role,
        'seed': seed,
        'variant': variant,
        'n_generations': len(gens),
        'n_claims_scored': int(len(vdf)),
        'conditions': list(settings.CONDITIONS),
        'condition_summary': sdf.to_dict('records'),
        'top_factual_unsupported_reasons': top_factual_reasons,
        'top_all_failure_reasons_excluding_caution': top_all_reasons,
        'primary_metrics': [
            'evidence_faithfulness',
            'diagnostic_temporal_accuracy',
            'relational_detection_accuracy',
        ],
        'control_metric': 'evidence_control_abstention_accuracy (C0 only)',
        'supplementary_metrics': [
            'relational_precision',
            'relational_set_recall',
            'relational_set_f1',
            'relational_abstention_correct_rate',
        ],
        'judge': 'deterministic (no LLM-as-judge)',
    }
    print(sdf.to_string(index=False), flush=True)
    print(
        '\n상위 factual 미지지 사유:',
        json.dumps(top_factual_reasons, indent=2, ensure_ascii=False),
        flush=True,
    )
    print(
        '\n상위 전체 실패 사유(caution 제외):',
        json.dumps(top_all_reasons, indent=2, ensure_ascii=False),
        flush=True,
    )
    settings.write_json(settings.MANIFESTS / f'evaluation_{role}_seed{seed}{tag}.json', summary)
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
    ap.add_argument('--variant', default='B5', choices=['B5'])
    ap.add_argument('--generations', type=Path, help='Existing generation JSONL to evaluate')
    args = ap.parse_args()
    main(args.role, args.seed, args.variant, args.generations)
