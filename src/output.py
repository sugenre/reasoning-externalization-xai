"""Structured output parsing and narrative export."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
import settings

SUPPORTED = {
    '$schema',
    'title',
    'description',
    'type',
    'required',
    'additionalProperties',
    'properties',
    'enum',
    'minimum',
    'maximum',
    'minLength',
    'maxLength',
    'minItems',
    'maxItems',
    'items',
}
_TYPES = {
    'object': dict,
    'array': list,
    'string': str,
    'integer': int,
    'number': (int, float),
    'boolean': bool,
    'null': type(None),
}


# Schema validation
class SchemaError(ValueError):
    pass


def _check_supported(schema: dict, path: str = '$') -> None:
    unknown = set(schema) - SUPPORTED
    if unknown:
        raise SchemaError(f'{path}: 지원하지 않는 스키마 키워드 {sorted(unknown)}')
    for k, v in schema.get('properties', {}).items():
        _check_supported(v, f'{path}.{k}')
    if isinstance(schema.get('items'), dict):
        _check_supported(schema['items'], f'{path}[]')


def _type_ok(value, t) -> bool:
    types = t if isinstance(t, list) else [t]
    for name in types:
        py = _TYPES[name]
        if name == 'integer':
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            return True
        if name == 'number' and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def validate(instance, schema: dict, path: str = '$') -> None:
    """Schema validation; violations raise SchemaError."""
    if path == '$':
        _check_supported(schema)
    if 'type' in schema and (not _type_ok(instance, schema['type'])):
        raise SchemaError(
            f"{path}: type 이 {schema['type']} 여야 하는데 {type(instance).__name__} 임"
        )
    if 'enum' in schema and instance not in schema['enum']:
        raise SchemaError(f"{path}: {instance!r} 은 허용 목록에 없음 ({schema['enum']})")
    if isinstance(instance, str):
        if 'minLength' in schema and len(instance) < schema['minLength']:
            raise SchemaError(f"{path}: 길이 {len(instance)} < {schema['minLength']}")
        if 'maxLength' in schema and len(instance) > schema['maxLength']:
            raise SchemaError(f"{path}: 길이 {len(instance)} > {schema['maxLength']}")
    if isinstance(instance, (int, float)) and (not isinstance(instance, bool)):
        if 'minimum' in schema and instance < schema['minimum']:
            raise SchemaError(f"{path}: {instance} < 최소 {schema['minimum']}")
        if 'maximum' in schema and instance > schema['maximum']:
            raise SchemaError(f"{path}: {instance} > 최대 {schema['maximum']}")
    if isinstance(instance, list):
        if 'minItems' in schema and len(instance) < schema['minItems']:
            raise SchemaError(f"{path}: 원소 {len(instance)} < {schema['minItems']}")
        if 'maxItems' in schema and len(instance) > schema['maxItems']:
            raise SchemaError(f"{path}: 원소 {len(instance)} > {schema['maxItems']}")
        if isinstance(schema.get('items'), dict):
            for i, v in enumerate(instance):
                validate(v, schema['items'], f'{path}[{i}]')
    if isinstance(instance, dict):
        for req in schema.get('required', []):
            if req not in instance:
                raise SchemaError(f'{path}: 필수 필드 {req!r} 가 없음')
        props = schema.get('properties', {})
        if schema.get('additionalProperties') is False:
            extra = set(instance) - set(props)
            if extra:
                raise SchemaError(f'{path}: 허용되지 않은 필드 {sorted(extra)}')
        for k, v in instance.items():
            if k in props:
                validate(v, props[k], f'{path}.{k}')


_FENCE = re.compile('```(?:json)?\\s*(.*?)```', re.DOTALL)


def load_schema(cfg: dict | None = None) -> dict:
    cfg = cfg if cfg is not None else settings.llm_config()
    path = cfg['generation']['schema_file']
    with open(path if str(path).startswith('/') else settings.ROOT / path) as fh:
        return json.load(fh)


def _scan_json(c: str, open_ch: str, close_ch: str):
    """First complete JSON value with the requested delimiters."""
    start = c.find(open_ch)
    if start < 0:
        return None
    depth, in_str, esc = (0, False, False)
    for i, ch in enumerate(c[start:], start=start):
        if in_str:
            if esc:
                esc = False
            elif ch == '\\':
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(c[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


# Output parsing
def extract_json(text: str) -> dict:
    """JSON extraction with code-fence and bare-claims-array handling."""
    m = _FENCE.search(text)
    candidates = ([m.group(1)] if m else []) + [text]
    for c in candidates:
        c = c.strip()
        arr = _scan_json(c, '[', ']')
        if isinstance(arr, list) and arr and all((isinstance(x, dict) for x in arr)):
            obj = _scan_json(c, '{', '}')
            if not isinstance(obj, dict) or 'claims' not in obj:
                if c.find('[') < (c.find('{') if '{' in c else len(c)):
                    return {'claims': arr}
        start = c.find('{')
        if start < 0:
            continue
        depth, in_str, esc = (0, False, False)
        for i, ch in enumerate(c[start:], start=start):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(c[start : i + 1])
                    except json.JSONDecodeError:
                        break
    raise SchemaError('완결된 JSON 객체를 찾지 못함')


def check_structural_rules(obj: dict, cfg: dict, *, require_factual: bool = True) -> None:
    """Claim budgets, citations, abstention fields, and null rank fields."""
    claims = obj['claims']
    factual = [c for c in claims if c['claim_type'] not in ('caution', 'abstain')]
    caution = [c for c in claims if c['claim_type'] == 'caution']
    abstain = [c for c in claims if c['claim_type'] == 'abstain']
    substantive = factual + abstain
    if len(substantive) > cfg['generation']['max_factual_sentences']:
        raise SchemaError(
            f"substantive 문장 {len(substantive)}개(사실 {len(factual)} + 기권 {len(abstain)}) "
            f"> 최대 {cfg['generation']['max_factual_sentences']}"
        )
    if require_factual and (not factual):
        raise SchemaError('사실 문장이 하나도 없음 (abstain·caution 만으로는 채점 불가)')
    if not require_factual and (not factual) and (not abstain):
        raise SchemaError('질문 답변에는 사실 주장 또는 명시적 abstain이 필요함')
    if len(caution) > cfg['generation']['max_caution_sentences']:
        raise SchemaError(
            f"주의 문장 {len(caution)}개 > 최대 {cfg['generation']['max_caution_sentences']}"
        )
    if len(abstain) > cfg['generation']['max_abstain_sentences']:
        raise SchemaError(
            f"기권 문장 {len(abstain)}개 > 최대 {cfg['generation']['max_abstain_sentences']}"
        )
    ids = [c['sentence_id'] for c in claims]
    if ids != list(range(1, len(ids) + 1)):
        raise SchemaError(f'sentence_id 가 1..n 연속이 아님: {ids}')
    for c in claims:
        if c.get('from_rank') is not None or c.get('to_rank') is not None:
            raise SchemaError(f"문장 {c['sentence_id']}: from_rank/to_rank 모두 null 지정 필요")
    for c in factual:
        if not c['evidence_ids']:
            raise SchemaError(f"사실 문장 {c['sentence_id']} 에 evidence_ids 가 없음")
    for c in abstain:
        if c.get('predicate') != 'no_supporting_evidence':
            raise SchemaError(
                f"abstain 문장 {c['sentence_id']}: predicate 는 no_supporting_evidence 지정 필요"
            )
        if c.get('value') is not None:
            raise SchemaError(f"abstain 문장 {c['sentence_id']}: value 는 null 지정 필요")
        if c.get('evidence_ids'):
            raise SchemaError(f"abstain 문장 {c['sentence_id']}: evidence_ids 는 [] 지정 필요")


# Narrative construction
def render_narrative(claims: list[dict]) -> str:
    """Claim texts in sentence order."""
    return ' '.join((c['text'].strip() for c in sorted(claims, key=lambda x: x['sentence_id'])))


def parse(text: str, schema: dict, cfg: dict, *, require_factual: bool = True) -> dict:
    obj = extract_json(text)
    obj.pop('narrative', None)
    validate(obj, schema)
    check_structural_rules(obj, cfg, require_factual=require_factual)
    obj = settings.canonical_names(obj)
    obj['narrative'] = render_narrative(obj['claims'])
    return obj


def try_parse(
    text: str, schema: dict, cfg: dict, *, require_factual: bool = True
) -> tuple[dict | None, str | None]:
    try:
        return (parse(text, schema, cfg, require_factual=require_factual), None)
    except (SchemaError, ValueError, KeyError, TypeError) as e:
        return (None, f'{type(e).__name__}: {e}')


OUTDIR = settings.ARTIFACTS / 'narratives'
CSV_COLS = ['case_id', 'month', 'condition', 'model_id', 'n_claims', 'n_chars', 'ok', 'narrative']


def _tag(path) -> str:
    return path.name[len('generations_') : -len('.jsonl')]


def _csv_cell(v) -> str:
    s = '' if v is None else str(v)
    return '"' + s.replace('"', '""') + '"' if any((c in s for c in ',"\n')) else s


# Narrative export
def export(path) -> dict:
    tag = _tag(path)
    rows, n, n_ok, n_fail, n_mismatch = ([], 0, 0, 0, 0)
    for line in open(path):
        d = json.loads(line)
        n += 1
        parsed = settings.canonical_names(d.get('parsed'))
        if not parsed or not parsed.get('claims'):
            rows.append(
                dict(
                    case_id=d['case_id'],
                    month=d['month'],
                    condition=d['condition'],
                    model_id=d.get('model_id'),
                    n_claims=0,
                    n_chars=0,
                    ok=False,
                    narrative='',
                    parse_error=d.get('parse_error'),
                )
            )
            n_fail += 1
            continue
        text = render_narrative(parsed['claims'])
        if parsed.get('narrative', '').strip() != text.strip():
            n_mismatch += 1
        rows.append(
            dict(
                case_id=d['case_id'],
                month=d['month'],
                condition=d['condition'],
                model_id=d.get('model_id'),
                n_claims=len(parsed['claims']),
                n_chars=len(text),
                ok=True,
                narrative=text,
                parse_error=None,
            )
        )
        n_ok += 1
    OUTDIR.mkdir(parents=True, exist_ok=True)
    with open(OUTDIR / f'narratives_{tag}.jsonl', 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    with open(OUTDIR / f'narratives_{tag}.csv', 'w') as fh:
        fh.write(','.join(CSV_COLS) + '\n')
        for r in rows:
            fh.write(','.join((_csv_cell(r[c]) for c in CSV_COLS)) + '\n')
    with open(OUTDIR / f'narratives_{tag}.txt', 'w') as fh:
        for cid in sorted({r['case_id'] for r in rows}):
            fh.write(f'=== {cid} ===\n')
            for r in [x for x in rows if x['case_id'] == cid]:
                body = r['narrative'] if r['ok'] else f"(파싱 실패: {r['parse_error']})"
                fh.write(f"  [{r['condition']}]\n    {body}\n")
            fh.write('\n')
    return dict(
        tag=tag, source=str(path), n=n, n_ok=n_ok, n_fail=n_fail, n_mismatch_vs_parsed=n_mismatch
    )


def main():
    want = sys.argv[1:] or None
    files = sorted(
        (
            p
            for p in settings.A_GENERATIONS.glob('generations_*.jsonl')
            if '_smoke' not in p.name and 'ABORTED' not in p.name
        )
    )
    if want:
        files = [p for p in files if _tag(p) in want]
    if not files:
        raise SystemExit(f'대상 파일이 없음. (요청 {want})')
    report = [export(p) for p in files]
    for r in report:
        flag = '' if r['n_mismatch_vs_parsed'] == 0 else f"  ★불일치 {r['n_mismatch_vs_parsed']}"
        print(
            f"{r['tag']:28s} n={r['n']:5d}  narrative 생성 {r['n_ok']:5d}  "
            f"파싱실패 {r['n_fail']:3d}{flag}",
            flush=True,
        )
    manifest = dict(
        step='09_export_narratives',
        rule='narrative = join(claims[].text, sep=" ", order=sentence_id asc)',
        rule_source='src/output.py::render_narrative',
        exported_at=datetime.now(timezone.utc).astimezone().isoformat(),
        files=report,
    )
    (OUTDIR / 'export_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'\n→ {OUTDIR}')


if __name__ == '__main__':
    main()
