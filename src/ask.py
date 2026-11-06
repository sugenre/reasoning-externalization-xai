"""PERMNO-based C3-Analog question answering."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date
import json
import os
from pathlib import Path
import re
import uuid
import duckdb
import database
import generation
import output
import prompts as prompt_templates
import scoring
import settings

QUESTION_RULES = (settings.ROOT / 'prompts' / 'question_answering.txt').read_text(encoding='utf-8')
CASE_PATTERN = re.compile('(?<![A-Za-z0-9_])(\\d{4}-\\d{2}_Q[1-5]_\\d+)(?![A-Za-z0-9_])')
PERMNO_PATTERN = re.compile(
    '(?:\\bPERMNO|종목(?:\\s*(?:식별번호|번호))?)\\s*[:=#]?\\s*(\\d{1,9})(?!\\d)', re.I
)
MONTH_PATTERN = re.compile(
    '(?<!\\d)(\\d{4})[-/](\\d{1,2})(?!\\d)|(?<!\\d)(\\d{4})\\s*년\\s*(\\d{1,2})\\s*월'
)


class QuestionInputError(ValueError):
    """Invalid or ambiguous stock-month selection."""


# Case selection
def normalize_month(value: str) -> str:
    try:
        year, month = (int(part) for part in value.split('-'))
        return date(year, month, 1).strftime('%Y-%m')
    except (ValueError, TypeError) as exc:
        raise QuestionInputError('기준월에 YYYY-MM 형식의 유효한 월 필요') from exc


def resolve_case(
    con: duckdb.DuckDBPyConnection,
    question: str,
    *,
    case_id: str | None = None,
    permno: int | None = None,
    month: str | None = None,
) -> dict:
    """Stock-month resolution with parameterized SQL."""
    case_ids = set(CASE_PATTERN.findall(question))
    if case_id is not None:
        case_ids.add(case_id)
    if len(case_ids) > 1:
        raise QuestionInputError('한 번에 하나의 case_id 지정 필요')
    selected_id = next(iter(case_ids), None)
    permnos = {int(value) for value in PERMNO_PATTERN.findall(question)}
    if permno is not None:
        if permno <= 0:
            raise QuestionInputError('PERMNO에 양의 정수 필요')
        permnos.add(permno)
    if len(permnos) > 1:
        raise QuestionInputError('질문과 옵션의 PERMNO 불일치 또는 여러 종목 지정')
    selected_permno = next(iter(permnos), None)
    selected_month = normalize_month(month) if month is not None else None
    if selected_month is None and selected_id is None:
        months = {
            normalize_month(f'{a or c}-{b or d}') for a, b, c, d in MONTH_PATTERN.findall(question)
        }
        if len(months) > 1:
            raise QuestionInputError('여러 월 언급됨. --month로 기준월 지정 필요')
        selected_month = next(iter(months), None)
    if selected_id is not None:
        matches = database.rows(
            con, 'SELECT case_id, PERMNO, month FROM cases WHERE case_id = ?', [selected_id]
        )
    else:
        if selected_permno is None or selected_month is None:
            raise QuestionInputError(
                (
                    'PERMNO와 기준월 필요. 질문에 "PERMNO 38850, 2020-08 기준" 입력 또는 --permno와'
                    ' --month 또는 --case-id 지정 필요'
                )
            )
        matches = database.rows(
            con,
            'SELECT case_id, PERMNO, month FROM cases WHERE PERMNO = ? AND month = ?',
            [selected_permno, selected_month],
        )
    if len(matches) != 1:
        raise QuestionInputError(
            '일치하는 DB 사례가 정확히 하나 필요. 현재 DB의 표본 범위 확인 필요'
        )
    selected = matches[0]
    if selected_permno is not None and int(selected['PERMNO']) != selected_permno:
        raise QuestionInputError('case_id와 PERMNO 불일치')
    if selected_month is not None and selected['month'] != selected_month:
        raise QuestionInputError('case_id와 기준월 불일치')
    return selected


# Question input
def prepare_question(
    con: duckdb.DuckDBPyConnection,
    question: str,
    *,
    case_id: str | None = None,
    permno: int | None = None,
    month: str | None = None,
    language: str = 'auto',
) -> dict:
    """C3-Analog evidence and question-specific instructions."""
    question = question.strip()
    if not question or len(question) > 2000:
        raise QuestionInputError('질문 길이 1~2,000자 필요')
    instructions = {
        'auto': 'Write claim text in the same language as the user question.',
        'ko': 'Write claim text in Korean.',
        'en': 'Write claim text in English.',
    }
    if language not in instructions:
        raise QuestionInputError('답변 언어에 auto, ko, en 중 하나 지정 필요')
    selected = resolve_case(con, question, case_id=case_id, permno=permno, month=month)
    year, basis_month = map(int, selected['month'].split('-'))
    target_month = f'{year + (basis_month == 12):04d}-{basis_month % 12 + 1:02d}'
    evidence_block = prompt_templates.render_evidence_block(
        con, selected['case_id'], settings.C3_ANALOG
    )
    selection = {
        'case_id': selected['case_id'],
        'PERMNO': int(selected['PERMNO']),
        'basis_month': selected['month'],
        'prediction_month': target_month,
    }
    user = (
        evidence_block
        + '\n\nSELECTION_JSON\n'
        + json.dumps(selection, ensure_ascii=False)
        + '\n\nUSER_QUESTION_JSON\n'
        + json.dumps({'question': question}, ensure_ascii=False)
        + '\n\nAnswer this question under the system rules. Return only the claims JSON object.'
    )
    return {
        'mode': 'question_answering',
        'condition': settings.C3_ANALOG,
        'question': question,
        'language': language,
        **selection,
        'month': selected['month'],
        'system': prompt_templates.SYSTEM_PROMPT
        + '\n\n'
        + QUESTION_RULES
        + '\n'
        + instructions[language],
        'user': user,
        'allowed_evidence_ids': prompt_templates.allowed_evidence_ids(
            con, selected['case_id'], settings.C3_ANALOG
        ),
    }


# Generation and verification
def answer_question(
    con: duckdb.DuckDBPyConnection,
    prepared: dict,
    runner: generation.VLLMRunner,
    cfg: dict | None = None,
) -> dict:
    """Answer generation, format repair, and factual-field verification."""
    cfg = cfg if cfg is not None else settings.llm_config()
    schema = output.load_schema(cfg)
    attempts = []
    request = prepared['user']
    parsed = None
    error = None
    context_limited = False
    for attempt in range(int(cfg['generation']['repair_retries']) + 1):
        tokenizer = getattr(runner, 'tok', None)
        if tokenizer is not None:
            rendered = runner.render(prepared['system'], request)
            input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
            context_limit = int(cfg['framework']['vllm_engine_args']['max_model_len'])
            if input_tokens + int(cfg['decoding']['max_new_tokens']) > context_limit:
                error = (
                    'Context limit: the question or format-repair prompt leaves '
                    'insufficient space for the configured answer. Shorten the '
                    'question.'
                )
                attempts.append({'attempt': attempt, 'raw_output': None, 'parse_error': error})
                context_limited = True
                break
        outputs = runner.generate([(prepared['system'], request)])
        if len(outputs) != 1:
            raise RuntimeError('Expected exactly one generated answer')
        raw = outputs[0]
        parsed, error = output.try_parse(raw, schema, cfg, require_factual=False)
        attempts.append({'attempt': attempt, 'raw_output': raw, 'parse_error': error})
        if parsed is not None:
            break
        request = prepared['user'] + '\n\n' + prompt_templates.repair_prompt(error, raw, cfg)
    result = {
        'mode': 'question_answering',
        'condition': settings.C3_ANALOG,
        'case_id': prepared['case_id'],
        'PERMNO': prepared['PERMNO'],
        'basis_month': prepared['basis_month'],
        'prediction_month': prepared['prediction_month'],
        'question': prepared['question'],
        'language': prepared['language'],
        'status': 'context_limit' if context_limited else 'invalid_output',
        'answer': None,
        'parsed': parsed,
        'parse_error': error,
        'n_repair_attempts': max(0, sum((a['raw_output'] is not None for a in attempts)) - 1),
        'attempts': attempts,
        'validation': None,
    }
    if parsed is None:
        return result
    evidence = database.CaseEvidence(con, prepared['case_id'])
    factual = [
        claim
        for claim in parsed['claims']
        if claim['claim_type'] in prompt_templates.FACTUAL_CLAIM_TYPES
    ]
    verdicts = [
        asdict(scoring.verify_claim(claim, evidence, settings.C3_ANALOG)) for claim in factual
    ]
    n_supported = sum((verdict['supported'] for verdict in verdicts))
    result['validation'] = {
        'n_factual_claims': len(factual),
        'n_supported': n_supported,
        'evidence_faithfulness': n_supported / len(factual) if factual else None,
        'verdicts': verdicts,
        'scope': (
            'Structured factual fields and citations only; prose meaning, '
            'question relevance, completeness, and correctness of abstention '
            'are not automatically verified.'
        ),
    }
    if n_supported != len(factual):
        result['status'] = 'unsupported_claims'
        return result
    abstained = any((claim['claim_type'] == 'abstain' for claim in parsed['claims']))
    result['status'] = 'abstained' if not factual else 'partial' if abstained else 'answered'
    result['answer'] = parsed['narrative']
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description='DB의 한 종목·기준월에 대해 C3-Analog 근거로 답변')
    ap.add_argument('--question', help='생략 시 터미널에서 질문 입력')
    ap.add_argument('--case-id')
    ap.add_argument('--permno', type=int)
    ap.add_argument('--month', help='예측 기준월 YYYY-MM; 예측 대상은 다음 달')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--language', choices=['auto', 'ko', 'en'], default='auto')
    ap.add_argument(
        '--role',
        default='primary',
        choices=[
            key
            for key, value in settings.llm_config().items()
            if isinstance(value, dict) and 'model_id' in value
        ],
    )
    ap.add_argument('--dry-run', action='store_true', help='모델 실행 없이 입력 프롬프트 출력')
    ap.add_argument(
        '--output',
        type=Path,
        help='비공개 답변 기록 경로; 기본값 OUTPUT_ROOT/questions',
    )
    args = ap.parse_args()
    try:
        question = args.question if args.question is not None else input('질문: ')
        with database.connect(args.seed) as con:
            prepared = prepare_question(
                con,
                question,
                case_id=args.case_id,
                permno=args.permno,
                month=args.month,
                language=args.language,
            )
            if args.dry_run:
                print(json.dumps(prepared, ensure_ascii=False, indent=2))
                return 0
            cfg = settings.llm_config()
            expected_device = cfg[args.role]['cuda_visible_devices']
            if os.environ.get('CUDA_VISIBLE_DEVICES') != expected_device:
                raise QuestionInputError(
                    f'{args.role} 실행에는 CUDA_VISIBLE_DEVICES={expected_device} 설정 필요'
                )
            runner = generation.VLLMRunner(args.role, cfg=cfg)
            result = answer_question(con, prepared, runner, cfg)
            result.update(
                {
                    'seed': args.seed,
                    'role': args.role,
                    'model_id': cfg[args.role]['model_id'],
                    'database': str(settings.database_path(args.seed)),
                    'runner': runner.info(),
                    'prompt': prepared,
                }
            )
        output_path = args.output or settings.OUTPUTS / 'questions' / f'{uuid.uuid4().hex}.json'
        settings.write_json(output_path, result)
        print(f"\n[{result['status']}] {result['case_id']}")
        if result['status'] == 'context_limit':
            print('질문 또는 형식 복구 입력이 모델의 문맥 한도 초과. 질문 축소 후 재실행 필요')
        elif result['answer'] is None:
            print('출력 형식 또는 구조화 근거 검증 실패. 기록에서 실패 사유 확인 필요')
        else:
            for claim in result['parsed']['claims']:
                citations = ' '.join((f'[{eid}]' for eid in claim['evidence_ids']))
                print(claim['text'] + (' ' + citations if citations else ''))
        print(f'기록: {output_path}')
        return 0 if result['answer'] is not None else 1
    except (
        QuestionInputError,
        generation.IncompatibleStack,
        duckdb.Error,
        FileNotFoundError,
    ) as exc:
        print(f'[STOP] {exc}')
        return 2
    except (EOFError, KeyboardInterrupt):
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
