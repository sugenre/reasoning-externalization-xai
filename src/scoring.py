"""Structured-claim verification and task diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, field
import database
import prompts as prompt_templates
import settings

_ND = settings.llm_config()['numeric_display']
TOL = {k: 1.5 * 0.5 * 10 ** (-v) for k, v in _ND.items()}


@dataclass
class ClaimVerdict:
    sentence_id: int
    claim_type: str
    supported: bool
    reason: str
    cited_ids: list[str] = field(default_factory=list)


def _tol(name: str) -> float:
    if name not in TOL:
        raise KeyError(
            f'numeric_display.{name} 가 없음. config/settings.yaml의 llm.numeric_display에 표시 자릿수 정의 필요'
        )
    return TOL[name]


def _num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace('%', '').strip())
        except ValueError:
            return None
    return None


def _close(a, b, tol: float) -> bool:
    va = _num(a)
    return va is not None and abs(va - float(b)) <= tol


def _temporal_gold(evidence: database.CaseEvidence, feature: str) -> dict | None:
    return evidence.features.get(feature)


def _overtakes_gold(evidence: database.CaseEvidence) -> set[tuple[str, str]]:
    return evidence.relation_pairs()


def _focal_features(evidence: database.CaseEvidence) -> list[str]:
    return evidence.focal_features()


# Claim verification
def verify_claim(claim: dict, evidence: database.CaseEvidence, condition: str) -> ClaimVerdict:
    """Factual-field and citation checks against condition-visible evidence."""
    claim = settings.canonical_names(claim)
    visible_ids = evidence.visible_ids(condition)
    sid = int(claim['sentence_id'])
    ct = claim['claim_type']
    cited = list(claim.get('evidence_ids') or [])

    def no(reason: str) -> ClaimVerdict:
        return ClaimVerdict(sid, ct, False, reason, cited)

    def yes(reason: str = 'ok') -> ClaimVerdict:
        return ClaimVerdict(sid, ct, True, reason, cited)

    if ct == 'caution':
        return yes('caution_not_factual')
    if ct == 'abstain':
        if claim.get('predicate') != 'no_supporting_evidence':
            return no('abstain_predicate_wrong')
        if claim.get('value') is not None or cited:
            return no('abstain_must_have_null_value_and_no_evidence_ids')
        if condition != settings.C0_EVIDENCE_CONTROL:
            return no('abstained_despite_temporal_evidence_available')
        if claim.get('subject') not in _focal_features(evidence):
            return no('abstain_wrong_diagnostic_target')
        return yes('correct_evidence_control_abstention')
    pred = claim.get('predicate')
    if pred not in prompt_templates.PREDICATES.get(ct, []):
        return no(f'predicate_not_allowed_for_claim_type:{pred}')
    if not cited:
        return no('no_evidence_cited')
    all_ids = evidence.registry
    for eid in cited:
        if eid not in visible_ids:
            return no(
                'cited_id_not_visible_in_condition' if eid in all_ids else 'cited_id_does_not_exist'
            )
    subj = claim.get('subject')
    val = claim.get('value')
    if ct == 'prediction':
        if subj != 'prediction':
            return no('prediction_subject_wrong')
        n = evidence.cited(cited, ('PREDICTION',))
        if n is None:
            return no('prediction_claim_must_cite_PRED')
        if val is None:
            return no('prediction_claim_requires_value')
        if pred == 'predicted_return':
            return (
                yes()
                if _close(val, n['pred'], _tol('ret'))
                else no('predicted_return_value_mismatch')
            )
        return yes() if _num(val) == float(n['quintile']) else no('quintile_mismatch')
    if ct == 'local_contribution':
        n = evidence.cited(cited, ('LOCAL_FEATURE',), subj)
        if n is None:
            return no('local_claim_must_cite_LOC_of_subject')
        phi = float(n['phi'])
        if phi == 0.0:
            return no('zero_local_shap_has_no_allowed_direction_predicate')
        want = 'contributes_positive' if phi > 0 else 'contributes_negative'
        if pred != want:
            return no('sign_of_local_shap_contradicted')
        if val is None:
            return no('local_shap_claim_requires_value')
        return yes() if _close(val, phi, _tol('phi')) else no('local_shap_value_mismatch')
    if ct == 'global_importance':
        n = evidence.cited(cited, ('GLOBAL_FEATURE',), subj)
        if n is None:
            return no('global_claim_must_cite_GLB_of_subject')
        if val is None:
            return no('global_importance_claim_requires_value')
        return yes() if _close(val, n['global_P'], _tol('P')) else no('global_P_value_mismatch')
    if ct in ('temporal_trend', 'temporal_change'):
        gold = _temporal_gold(evidence, subj)
        if gold is None:
            return no('subject_not_a_displayed_feature')
        if condition == settings.C1_RAW:
            source = evidence.cited(cited, ('TEMPORAL_SEQUENCE',), subj)
            if source is None:
                return no('C1_temporal_claim_must_cite_SEQ_of_subject')
        elif condition in (settings.C2_NUMERICAL, settings.C3_RELATIONAL, settings.C3_ANALOG):
            source = evidence.cited(cited, ('TEMPORAL_NUMERICAL',), subj)
            if source is None:
                return no('C2plus_temporal_claim_must_cite_TMP_of_subject')
        else:
            return no('temporal_claim_not_supported_in_C0')
        if ct == 'temporal_trend':
            if val is not None:
                return no('temporal_trend_value_must_be_null')
            return yes() if pred == gold['trend'] else no('trend_direction_wrong')
        if val is None:
            return no('temporal_change_requires_value')
        return (
            yes() if _close(val, gold['delta6P'], _tol('delta6P')) else no('delta6P_value_mismatch')
        )
    if ct == 'relation_overtakes':
        if not isinstance(subj, str) or not isinstance(val, str):
            return no('overtakes_requires_feature_subject_and_string_value')
        pair = (subj, val)
        if pair not in _overtakes_gold(evidence):
            return no('overtakes_relation_not_in_gold')
        if condition in (settings.C1_RAW, settings.C2_NUMERICAL):
            needed = {f'{evidence.case_id}#SEQ:{subj}', f'{evidence.case_id}#SEQ:{val}'}
            if not needed.issubset(set(cited)):
                return no('raw_derived_overtakes_must_cite_both_SEQ_ids')
            return yes()
        if condition in (settings.C3_RELATIONAL, settings.C3_ANALOG):
            relation = next(
                (
                    row
                    for row in evidence.relations
                    if row['evidence_id'] in cited
                    and row['overtaker'] == subj
                    and (row['overtaken'] == val)
                ),
                None,
            )
            return (
                yes() if relation is not None else no('supplied_overtakes_must_cite_relation_edge')
            )
        return no('overtakes_not_supported_in_C0')
    if subj != 'historical_analogs' and ct.startswith('analog_'):
        return no('analog_subject_must_be_historical_analogs')
    if ct == 'analog_similarity':
        a = evidence.cited(cited, ('HISTORICAL_ANALOG', 'SIMILAR_TO'))
        if a is None:
            return no('analog_similarity_must_cite_ANALOG_or_SIMILAR')
        if val is None:
            return no('analog_similarity_requires_distance_value')
        return (
            yes()
            if _close(val, a['euclidean_distance'], _tol('distance'))
            else no('distance_value_mismatch')
        )
    if ct == 'analog_universe_outcome':
        o = evidence.cited(cited, ('ANALOG_UNIVERSE_OUTCOME',))
        if o is None:
            return no('universe_outcome_claim_must_cite_ANALOG_UNIVERSE_OUTCOME')
        if val is None:
            return no('universe_outcome_requires_value')
        return (
            yes()
            if _close(val, o['subsequent_universe_return'], _tol('ret'))
            else no('subsequent_universe_return_value_mismatch')
        )
    if ct == 'analog_model_performance':
        p = evidence.cited(cited, ('ANALOG_MODEL_PERFORMANCE',))
        if p is None:
            return no('model_performance_claim_must_cite_ANALOG_MODEL_PERFORMANCE')
        if val is None:
            return no('monthly_rank_ic_requires_value')
        return (
            yes()
            if _close(val, p['monthly_rank_ic'], _tol('ic'))
            else no('monthly_rank_ic_value_mismatch')
        )
    if ct == 'analog_summary':
        s = evidence.cited(cited, ('ANALOG_SUMMARY',))
        if s is None:
            return no('analog_summary_claim_must_cite_ANALOGSUM')
        if val is None:
            return no('analog_summary_requires_value')
        if pred not in s:
            return no(f'analog_summary_predicate_missing_from_gold:{pred}')
        if pred.endswith('_count'):
            return yes() if _num(val) == float(s[pred]) else no(f'{pred}_value_mismatch')
        tol = _tol('ic') if 'rank_ic' in pred else _tol('ret')
        return yes() if _close(val, s[pred], tol) else no(f'{pred}_value_mismatch')
    return no(f'unhandled_claim_type:{ct}')


# Evidence faithfulness
def evidence_faithfulness(parsed: dict, evidence: database.CaseEvidence, condition: str) -> dict:
    verdicts = [verify_claim(c, evidence, condition) for c in parsed['claims']]
    factual = [v for v in verdicts if v.claim_type in prompt_templates.FACTUAL_CLAIM_TYPES]
    n = len(factual)
    k = sum((1 for v in factual if v.supported))
    return {
        'n_factual_claims': n,
        'n_supported': k,
        'evidence_faithfulness': k / n if n else float('nan'),
        'verdicts': [v.__dict__ for v in verdicts],
        'unsupported_reasons': [v.reason for v in factual if not v.supported],
    }


# Temporal diagnostic
def focal_temporal_diagnostic(
    parsed: dict, evidence: database.CaseEvidence, condition: str
) -> dict:
    parsed = settings.canonical_names(parsed)
    focals = set(_focal_features(evidence))
    gold_trends = {f: _temporal_gold(evidence, f)['trend'] for f in focals}
    trend_claims = [
        c
        for c in parsed['claims']
        if c.get('claim_type') == 'temporal_trend' and c.get('subject') in focals
    ]
    abstain_claims = [
        c
        for c in parsed['claims']
        if c.get('claim_type') == 'abstain' and c.get('subject') in focals
    ]
    if condition == settings.C0_EVIDENCE_CONTROL:
        correct = len(abstain_claims) >= 1 and len(trend_claims) == 0
        mode = 'abstain'
    else:
        correct_trend = any(
            (c.get('predicate') == gold_trends[c.get('subject')] for c in trend_claims)
        )
        wrong_trend = any(
            (c.get('predicate') != gold_trends[c.get('subject')] for c in trend_claims)
        )
        correct = correct_trend and (not wrong_trend) and (len(abstain_claims) == 0)
        mode = 'report_trend'
    return {
        'focal_features': sorted(focals),
        'expected_mode': mode,
        'gold_trends': gold_trends,
        'n_focal_trend_claims': len(trend_claims),
        'n_focal_abstain_claims': len(abstain_claims),
        'diagnostic_temporal_accuracy': float(bool(correct)),
    }


# Relational diagnostic
def relational_sets(parsed: dict) -> set[tuple[str, str]]:
    parsed = settings.canonical_names(parsed)
    out = set()
    for c in parsed['claims']:
        if (
            c.get('claim_type') == 'relation_overtakes'
            and isinstance(c.get('subject'), str)
            and isinstance(c.get('value'), str)
        ):
            out.add((c['subject'], c['value']))
    return out


def prf(pred_set: set, gold_set: set) -> dict:
    tp = len(pred_set & gold_set)
    p = tp / len(pred_set) if pred_set else float('nan')
    r = tp / len(gold_set) if gold_set else float('nan')
    f1 = (
        2 * p * r / (p + r)
        if p == p and r == r and (p + r > 0)
        else 0.0 if pred_set or gold_set else float('nan')
    )
    return {
        'tp': tp,
        'n_pred': len(pred_set),
        'n_gold': len(gold_set),
        'precision': p,
        'recall': r,
        'f1': f1,
    }


def relational_diagnostic(parsed: dict, evidence: database.CaseEvidence, condition: str) -> dict:
    """OVERTAKES detection; undefined for C0 without temporal evidence."""
    if condition == settings.C0_EVIDENCE_CONTROL:
        return {
            'relational_detection_accuracy': float('nan'),
            'relational_precision': float('nan'),
            'relational_set_recall': float('nan'),
            'relational_set_f1': float('nan'),
            'n_relational_pred': float('nan'),
            'n_relational_gold': float('nan'),
            'n_relational_tp': float('nan'),
            'n_relational_fp': float('nan'),
            'relational_gold_empty': float('nan'),
            'relational_abstained': float('nan'),
        }
    pred = relational_sets(parsed)
    gold = _overtakes_gold(evidence)
    tp = pred & gold
    fp = pred - gold
    if gold:
        case_correct = bool(tp) and (not fp)
    else:
        case_correct = not pred
    full_set = prf(pred, gold)
    return {
        'relational_detection_accuracy': float(bool(case_correct)),
        'relational_precision': full_set['precision'],
        'relational_set_recall': full_set['recall'],
        'relational_set_f1': full_set['f1'],
        'n_relational_pred': len(pred),
        'n_relational_gold': len(gold),
        'n_relational_tp': len(tp),
        'n_relational_fp': len(fp),
        'relational_gold_empty': len(gold) == 0,
        'relational_abstained': len(pred) == 0,
    }


def task_metrics(parsed: dict, evidence: database.CaseEvidence, condition: str) -> dict:
    """Focal-trend and relation-detection diagnostics."""
    t = focal_temporal_diagnostic(parsed, evidence, condition)
    r = relational_diagnostic(parsed, evidence, condition)
    return {**t, **r}
