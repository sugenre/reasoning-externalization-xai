"""Condition-specific prompts from SQL evidence."""

from __future__ import annotations

import duckdb
import pandas as pd
import argparse
from contextlib import closing
import hashlib
import json
import os
import uuid
import database
import settings

PREDICATES = {
    'prediction': ['predicted_return', 'prediction_quintile'],
    'local_contribution': ['contributes_positive', 'contributes_negative'],
    'global_importance': ['relative_reliance'],
    'temporal_trend': ['increasing', 'decreasing', 'flat'],
    'temporal_change': ['window_change_delta6P'],
    'relation_overtakes': ['overtakes'],
    'analog_similarity': ['similar_to'],
    'analog_universe_outcome': ['subsequent_universe_return'],
    'analog_model_performance': ['monthly_rank_ic'],
    'analog_summary': [
        'positive_universe_return_count',
        'negative_universe_return_count',
        'mean_subsequent_universe_return',
        'median_subsequent_universe_return',
        'mean_rank_ic',
        'median_rank_ic',
    ],
    'abstain': ['no_supporting_evidence'],
    'caution': ['caution'],
}
FACTUAL_CLAIM_TYPES = [k for k in PREDICATES if k not in ('abstain', 'caution')]
SYSTEM_PROMPT = (settings.ROOT / 'prompts' / 'generation_main.txt').read_text(encoding='utf-8')
USER_TEMPLATE = """{evidence_block}

TASK
Write a compact evidence-grounded explanation. Include at least one supported contemporaneous
factual claim about the prediction or a local contribution. Required diagnostic: for the
displayed feature with the largest absolute local_shap_signed, report its six-month
global-reliance trend; if that trend cannot be determined from supplied evidence, emit one
abstain claim. If an OVERTAKES relation among displayed features is supported, report it with a
relation_overtakes claim. If NO OVERTAKES relation is supported, write no relation claim at
all: silence is the correct answer. Never write a relation_overtakes claim to state that no
relation exists - a relation_overtakes claim always needs a real relation and its
evidence_id. If you want to say the relation could not be determined, use one abstain claim
instead.

You do NOT need to mention every displayed feature. Prioritize only the claims needed to
answer the task within the claim budget.

Return only the JSON object required by the system message.
"""
REPAIR_TEMPLATE = """Your previous reply was not valid JSON matching the required schema.
Error: {error}
Previous reply:
{previous}
Return the SAME content corrected to one valid JSON object. Do not add, remove, or change
factual content, numbers, or evidence_ids.

Common causes and their fixes, all of which are FORMAT-ONLY. Apply the arithmetic literally:
- "허용되지 않은 필드 ['narrative']": delete the "narrative" field entirely. The pipeline
  builds the narrative from your claim "text" fields; you must not emit one.
- "필수 필드 'claims' 가 없음": the top level must be an OBJECT {{"claims":[...]}}, not a bare
  array. Wrap your list of claims in {{"claims": ...}}.
- "substantive": let S = (claims whose claim_type is neither "caution" nor "abstain")
  + (claims whose claim_type is "abstain"). S must be <= {max_substantive}.
  DELETE whole claims from the end until S fits. Do not shorten text, do not merge claims.
  A {max_total}th claim is allowed ONLY if it is a caution claim; with no caution claim the
  array holds AT MOST {max_substantive} claims in total.
- "기권 문장"/"abstain": keep AT MOST {max_abstain} abstain claims - one for the required
  trend diagnostic and, if applicable, one for the OVERTAKES relation. DELETE every other
  abstain claim. Do not convert an abstain into another claim type. If evidence for
  something else is missing, simply do not write a claim about it.
- "주의 문장"/"caution": keep AT MOST {max_caution} caution claim.
- "claims: 원소"/"maxItems": delete whole claims from the end until the array fits
  ({max_total} at the very most).
- "predicate ... 은 허용 목록에 없음": the predicate must be one of the allowed VALUES for
  that claim_type, never an evidence field name. For temporal_trend use exactly
  "increasing", "decreasing" or "flat" - never "trend_direction".
- "사실 문장 ... evidence_ids 가 없음": every factual claim needs at least one evidence_id
  from the EVIDENCE block. An abstain claim keeps evidence_ids empty. If the offending claim
  is a relation_overtakes claim asserting that no relation exists, DELETE that claim
  entirely - stating the absence of a relation is not a claim in this schema.
- "sentence_id": renumber the claims 1..n in array order.
- missing keys: add the missing key with null (or [] for evidence_ids).

After fixing, recount every limit before answering.

Return JSON only.
"""


# Format repair
def repair_prompt(error: str, previous: str, cfg: dict) -> str:
    """Format-repair instructions with configured claim limits."""
    g = cfg['generation']
    b = int(g['max_factual_sentences'])
    return REPAIR_TEMPLATE.format(
        error=error,
        previous=previous,
        max_substantive=b,
        max_abstain=int(g['max_abstain_sentences']),
        max_caution=int(g['max_caution_sentences']),
        max_total=b + int(g['max_caution_sentences']),
    )


T_FEATURES_HEADER = (
    '[FEATURES SHOWN]  D = top-3 |local SHAP| union top-3 current '
    'global importance; narrative reasoning is restricted to D; analog '
    'retrieval uses all 30 features externally.'
)
T_LOCAL_HEADER = '[LOCAL SIGNED SHAP]  directional contribution to THIS stock prediction'
T_GLOBAL_HEADER = (
    '[GLOBAL IMPORTANCE P]  current relative model reliance across all '
    'stocks; NOT directional; all 30 P values sum to 1'
)
T_SEQ_HEADER = '[RAW GLOBAL IMPORTANCE SEQUENCE]  six monthly P values, oldest first'
T_NUM_HEADER = (
    '[NUMERICAL TEMPORAL EVIDENCE]  deterministic pipeline outputs over'
    ' the same 6 months; Delta6P=P_last-P_first; slope=OLS over all six'
    ' P values'
)
T_REL_HEADER = (
    '[RELATIONAL EVIDENCE]  deterministic endpoint crossing among '
    'displayed features: OVERTAKES(A,B) iff P_A(first)<P_B(first) and '
    'P_A(last)>P_B(last)'
)
T_REL_NONE = '  none: no OVERTAKES relation holds among displayed features.'
T_TRAJ_HEADER = '[CURRENT ATTRIBUTION TRAJECTORY]'
T_TRAJ_REPR = 'delta_P over L=6 months, 30 features, flattened to R^180'
T_TRAJ_DIST = 'euclidean'
T_ANALOG_HEADER = (
    '[HISTORICAL ANALOGS]  five earlier 30-feature ΔP trajectories; '
    'historical context only, not causal evidence and not a forecast.'
)
T_SUM_HEADER = '[ANALOG SUMMARY]  deterministic aggregate over K=5; do not recompute'
Q_CASE = (
    'SELECT month, PERMNO, pred, quintile, pred_pct_in_month, '
    'n_stocks_in_month, traj_window_start, traj_window_end FROM cases '
    'WHERE case_id = ?'
)
Q_FEATURES = """
SELECT e.feature, e.phi, e.phi_direction, e.val_raw, e.val_z, e.val_pct,
       e.global_P, e.seq_P, e.seq_months, e.delta6P, e.slope, e.trend,
       e.eid_loc, e.eid_glb, e.eid_seq, e.eid_tmp,
       m.name_en, m.formula_text, m.window_months, m.higher_means_en, m.discrete, m.caveat_en
FROM evidence e JOIN feature_metadata m USING (feature)
WHERE e.case_id = ? ORDER BY e.d_position
"""
Q_RELATIONS = """
SELECT r.overtaker, r.overtaken, r.start_P_overtaker, r.start_P_overtaken,
       r.end_P_overtaker, r.end_P_overtaken, r.evidence_id
FROM relations r
JOIN evidence a ON a.case_id = r.case_id AND a.feature = r.overtaker
JOIN evidence b ON b.case_id = r.case_id AND b.feature = r.overtaken
WHERE r.case_id = ? ORDER BY a.d_position, b.d_position
"""
Q_ANALOGS = """
SELECT analog_rank, analog_start, analog_end, euclidean_distance,
       return_month, subsequent_universe_return, monthly_rank_ic,
       eid_analog, eid_universe, eid_perf
FROM analogs WHERE case_id = ? ORDER BY analog_rank
"""
Q_SUMMARY = """
SELECT positive_universe_return_count, negative_universe_return_count,
       zero_universe_return_count, mean_subsequent_universe_return,
       median_subsequent_universe_return, mean_rank_ic, median_rank_ic, evidence_id
FROM analog_summary WHERE case_id = ?
"""
Q_ALLOWED = (
    'SELECT evidence_id FROM evidence_registry WHERE case_id = ? AND '
    'list_contains(visible_in, ?)'
)


def _fmt(x, nd: int) -> str:
    return f'{float(x):.{nd}f}'


def _describe(r: dict) -> str:
    """Feature description with explicit handling of missing metadata."""
    bits = [r['name_en'], f"formula: {r['formula_text']}"]
    if pd.notna(r['window_months']):
        bits.append(f"lookback: {int(r['window_months'])}m")
    bits.append(f"higher means: {r['higher_means_en']}")
    if r['discrete']:
        bits.append('DISCRETE: -1 or +1 only')
    if pd.notna(r['caveat_en']) and r['caveat_en']:
        bits.append(f"CAVEAT: {r['caveat_en']}")
    return '; '.join(bits)


# SQL evidence rendering
def render_evidence_block(
    con: duckdb.DuckDBPyConnection, case_id: str, condition: str, nd: dict | None = None
) -> str:
    """Condition-visible evidence text."""
    if condition not in settings.VISIBILITY['base']:
        raise ValueError(f'알 수 없는 condition: {condition}')
    nd = nd or settings.llm_config()['numeric_display']
    c = con.execute(Q_CASE, [case_id]).fetchone()
    if c is None:
        raise KeyError(f'DB 에 사례가 없음: {case_id}')
    month, permno, pred, quintile, pred_pct, n_stocks, w_start, w_end = c
    F = con.execute(Q_FEATURES, [case_id]).df().to_dict('records')
    if not F:
        raise AssertionError(f'{case_id}: evidence 행이 없음')

    def visible(layer: str) -> bool:
        return condition in settings.VISIBILITY[layer]

    L = [
        'EVIDENCE',
        f'stock_month: PERMNO {int(permno)}, month {month}',
        '',
        '[PREDICTION]',
        f"  predicted_next_month_return={_fmt(pred, nd['ret'])}; "
        f"prediction_quintile={int(quintile)}/5; "
        f"cross_sectional_percentile={int(pred_pct)}/99; "
        f"n_stocks={int(n_stocks)}; evidence_id={case_id}#PRED",
        '',
        T_FEATURES_HEADER,
    ]
    for r in F:
        L.append(f"  - {r['feature']}: {_describe(r)}")
    L.append('')
    L.append(T_LOCAL_HEADER)
    for r in F:
        L.append(
            f"  {r['feature']}: local_shap_signed={_fmt(r['phi'], nd['phi'])} "
            f"({r['phi_direction']}); raw={_fmt(r['val_raw'], nd['raw'])}; "
            f"z={_fmt(r['val_z'], nd['z'])}; percentile={int(r['val_pct'])}/99; "
            f"evidence_id={r['eid_loc']}"
        )
    L.append('')
    L.append(T_GLOBAL_HEADER)
    for r in F:
        L.append(
            f"  {r['feature']}: current_importance_P={_fmt(r['global_P'], nd['P'])}; "
            f"evidence_id={r['eid_glb']}"
        )
    L.append('')
    if visible('raw'):
        L.append(T_SEQ_HEADER)
        L.append('  months=' + ','.join(F[0]['seq_months']))
        for r in F:
            vals = ','.join((_fmt(x, nd['P']) for x in r['seq_P']))
            L.append(f"  {r['feature']}: P=[{vals}]; evidence_id={r['eid_seq']}")
        L.append('')
    if visible('numerical'):
        L.append(T_NUM_HEADER)
        for r in F:
            L.append(
                f"  {r['feature']}: window_change_delta6P={_fmt(r['delta6P'], nd['delta6P'])}; "
                f"trend_slope={_fmt(r['slope'], nd['slope'])} per month; "
                f"trend_direction={r['trend']}; evidence_id={r['eid_tmp']}"
            )
        L.append('')
    if visible('relational'):
        L.append(T_REL_HEADER)
        rels = con.execute(Q_RELATIONS, [case_id]).fetchall()
        if not rels:
            L.append(T_REL_NONE)
        for a, b, sa, sb, ea, eb, eid in rels:
            L.append(
                f"  OVERTAKES({a}>{b}); start_P=({_fmt(sa, nd['P'])},{_fmt(sb, nd['P'])}); "
                f"end_P=({_fmt(ea, nd['P'])},{_fmt(eb, nd['P'])}); evidence_id={eid}"
            )
        L.append('')
    if visible('analog'):
        L.extend(_analog_lines(con, case_id, w_start, w_end, nd))
    return settings.canonical_names('\n'.join(L).rstrip())


def _analog_lines(con, case_id: str, w_start: str, w_end: str, nd: dict) -> list[str]:
    L = [
        T_TRAJ_HEADER,
        f'  window={w_start}..{w_end}; representation={T_TRAJ_REPR}; '
        f'distance={T_TRAJ_DIST}; evidence_id={case_id}#TRAJ',
        '',
        T_ANALOG_HEADER,
    ]
    rows = con.execute(Q_ANALOGS, [case_id]).fetchall()
    if len(rows) != 5:
        raise AssertionError(f'{case_id}: C3-Analog 은 정확히 K=5 가 필요함 ({len(rows)}행)')
    for k, a_start, a_end, dist, ret_month, uret, ic, eid_a, eid_u, eid_p in rows:
        L.append(
            f"  rank{int(k)}: window={a_start}..{a_end}; "
            f"euclidean_distance={_fmt(dist, nd['distance'])}; evidence_id={eid_a}"
        )
        L.append(
            f"    subsequent_universe_return[{ret_month}]={_fmt(uret, nd['ret'])}; "
            f"evidence_id={eid_u}"
        )
        L.append(
            f"    monthly_rank_ic[prediction_month={a_end}, realized_month={ret_month}]="
            f"{_fmt(ic, nd['ic'])}; evidence_id={eid_p}"
        )
    s = con.execute(Q_SUMMARY, [case_id]).fetchone()
    if s is None:
        raise AssertionError(f'{case_id}: analog_summary 없음')
    pos, neg, zero, mean_u, med_u, mean_ic, med_ic, eid_s = s
    L.extend(
        [
            T_SUM_HEADER,
            f'  universe_return_counts: positive={int(pos)}; negative={int(neg)}; zero={int(zero)}',
            f"  mean_subsequent_universe_return={_fmt(mean_u, nd['ret'])}; "
            f"median_subsequent_universe_return={_fmt(med_u, nd['ret'])}",
            f"  mean_rank_ic={_fmt(mean_ic, nd['ic'])}; "
            f"median_rank_ic={_fmt(med_ic, nd['ic'])}; evidence_id={eid_s}",
            '',
        ]
    )
    return L


def allowed_evidence_ids(con, case_id: str, condition: str) -> list[str]:
    """Condition-visible identifiers in deterministic order."""
    ids = [r[0] for r in con.execute(Q_ALLOWED, [case_id, condition]).fetchall()]
    return sorted(settings.canonical_names(ids))


def build_prompt(con, case_id: str, condition: str) -> dict:
    """Evidence prompt and permitted identifiers."""
    month = con.execute('SELECT month FROM cases WHERE case_id = ?', [case_id]).fetchone()[0]
    block = render_evidence_block(con, case_id, condition)
    return {
        'case_id': case_id,
        'month': month,
        'condition': condition,
        'system': SYSTEM_PROMPT,
        'user': USER_TEMPLATE.format(evidence_block=block),
        'allowed_evidence_ids': allowed_evidence_ids(con, case_id, condition),
    }


# Condition boundaries
def check_no_leakage(block: str, condition: str) -> None:
    if condition not in settings.CONDITIONS:
        raise ValueError(f'Unknown condition: {condition}')
    for required in (
        '[PREDICTION]',
        '[FEATURES SHOWN]',
        '[LOCAL SIGNED SHAP]',
        '[GLOBAL IMPORTANCE P]',
    ):
        if required not in block:
            raise AssertionError(f'Missing evidence section: {required}')
    sections = {
        'raw': ['[RAW GLOBAL IMPORTANCE SEQUENCE]'],
        'numerical': ['[NUMERICAL TEMPORAL EVIDENCE]'],
        'relational': ['[RELATIONAL EVIDENCE]'],
        'analog': ['[CURRENT ATTRIBUTION TRAJECTORY]', '[HISTORICAL ANALOGS]', '[ANALOG SUMMARY]'],
    }
    for layer, markers in sections.items():
        for marker in markers:
            if (marker in block) != (condition in settings.VISIBILITY[layer]):
                raise AssertionError(f'{condition}: incorrect availability of {marker}')
    for marker in (
        'positive_rank_ic_count',
        'negative_rank_ic_count',
        'zero_rank_ic_count',
        'rank_ic_counts:',
    ):
        if marker in block:
            raise AssertionError(f'Undisplayed summary field in prompt: {marker}')


# Batch prompts
def main(seed: int = 42) -> dict:
    settings.A_PROMPTS.mkdir(parents=True, exist_ok=True)
    destination = settings.A_PROMPTS / f'prompts_seed{seed}.jsonl'
    temporary = destination.with_name(f'.{destination.name}.{uuid.uuid4().hex}')
    counts = {condition: 0 for condition in settings.CONDITIONS}
    try:
        with closing(database.connect(seed)) as con, temporary.open(
            'w', encoding='utf-8'
        ) as handle:
            database.validate_database(con)
            cases = con.execute(
                'SELECT case_id FROM cases ORDER BY month, quintile, PERMNO'
            ).fetchall()
            for (case_id,) in cases:
                for condition in settings.CONDITIONS:
                    prompt = build_prompt(con, case_id, condition)
                    check_no_leakage(
                        prompt['user'].split(
                            '\n\nTASK\n',
                            1,
                        )[0],
                        condition,
                    )
                    handle.write(json.dumps(prompt, ensure_ascii=False) + '\n')
                    counts[condition] += 1
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    summary = {
        'seed': seed,
        'n_cases': len(cases),
        'n_prompts': sum(counts.values()),
        'per_condition': counts,
        'system_prompt_sha256': hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        'prompts': str(destination),
    }
    settings.write_json(settings.MANIFESTS / f'build_conditions_seed{seed}.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    main(parser.parse_args().seed)
