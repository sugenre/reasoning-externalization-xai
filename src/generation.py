"""Configured LLM loading and batch generation."""

from __future__ import annotations

import importlib
import json
import os
import time
from dataclasses import dataclass
import argparse
import uuid
import output
import prompts as prompt_templates
import settings


class IncompatibleStack(RuntimeError):
    """Unavailable vLLM runtime."""


@dataclass
class RunnerInfo:
    framework: str
    versions: dict


# Runtime checks
def collect_versions() -> dict:
    import importlib.metadata as md

    out = {}
    for pkg in (
        'vllm',
        'torch',
        'transformers',
        'tokenizers',
        'xgboost',
        'numpy',
        'pandas',
        'pyarrow',
        'scipy',
    ):
        try:
            out[pkg] = md.version(pkg)
        except Exception:
            out[pkg] = None
    try:
        import torch

        out['torch_cuda'] = torch.version.cuda
        out['torch_cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            out['gpu_name'] = torch.cuda.get_device_name(0)
            out['gpu_capability'] = '.'.join(map(str, torch.cuda.get_device_capability(0)))
    except Exception as e:
        out['torch_error'] = str(e)
    out['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    return out


def require_vllm() -> None:
    """vLLM availability check."""
    try:
        importlib.import_module('vllm')
    except Exception as e:
        raise IncompatibleStack(
            f'vLLM import 불가.\n  원인: {type(e).__name__}: {e}\n  '
            f'requirements.txt와 CUDA 환경 확인 필요'
        ) from e


def assert_correct_gpu(model_cfg: dict) -> None:
    """Physical GPU assignment with PCI bus ordering."""
    want = model_cfg.get('expected_gpu_name_contains')
    if not want:
        return
    order = os.environ.get('CUDA_DEVICE_ORDER')
    if order != 'PCI_BUS_ID':
        raise AssertionError(
            f'CUDA_DEVICE_ORDER={order!r} 상태. 이종 GPU 환경에서 '
            f'nvidia-smi 인덱스 일치를 위한 PCI_BUS_ID 고정 필요. '
            f'export CUDA_DEVICE_ORDER=PCI_BUS_ID 설정 필요'
        )
    import torch

    if not torch.cuda.is_available():
        raise AssertionError('CUDA 사용 불가')
    name = torch.cuda.get_device_name(0)
    if want not in name:
        raise AssertionError(
            f"GPU 배치 오류: 기대 {want!r} 인데 보이는 장치는 {name!r} 임. "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r}"
        )


# LLM runner
class VLLMRunner:
    """Batched generation with prefix caching."""

    def __init__(self, role: str, cfg: dict | None = None):
        cfg = cfg if cfg is not None else settings.llm_config()
        self.cfg = cfg
        self.role = role
        self.model_cfg = cfg[role]
        self.dec = cfg['decoding']
        require_vllm()
        assert_correct_gpu(self.model_cfg)
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        eng = dict(cfg['framework']['vllm_engine_args'])
        eng['tensor_parallel_size'] = self.model_cfg['tensor_parallel_size']
        if eng['tensor_parallel_size'] != 1:
            raise AssertionError('텐서 병렬 사용 불가 (the configured execution setup)')
        eng['dtype'] = self.model_cfg['dtype']
        t0 = time.time()
        self.llm = LLM(model=self.model_cfg['model_id'], seed=self.dec['seed'], **eng)
        self.tok = AutoTokenizer.from_pretrained(self.model_cfg['model_id'])
        self.load_sec = round(time.time() - t0, 1)
        self.sampling = SamplingParams(
            temperature=self.dec['temperature'],
            top_p=self.dec['top_p'],
            top_k=self.dec['top_k'],
            repetition_penalty=self.dec['repetition_penalty'],
            max_tokens=self.dec['max_new_tokens'],
            seed=self.dec['seed'],
            n=1,
        )
        if self.dec['do_sample'] is not False or self.dec['temperature'] != 0.0:
            raise AssertionError('디코딩이 greedy가 아님 — 재현성 설정 위반')

    def revision(self) -> str | None:
        try:
            from huggingface_hub import HfApi

            return HfApi().model_info(self.model_cfg['model_id']).sha
        except Exception:
            return None

    def prompt_style(self) -> str:
        """Chat formatting by default; raw formatting for base models."""
        return self.model_cfg.get('prompt_style', 'chat')

    def render(self, system: str, user: str) -> str:
        if self.prompt_style() == 'raw':
            return system + '\n\n' + user + '\n\n'
        return self.tok.apply_chat_template(
            [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.model_cfg['thinking_mode'],
        )

    def generate(self, prompts: list[tuple[str, str]]) -> list[str]:
        texts = [self.render(s, u) for s, u in prompts]
        outs = self.llm.generate(texts, self.sampling)
        return [o.outputs[0].text for o in outs]

    def info(self) -> dict:
        import torch

        return {
            'framework': 'vllm',
            'role': self.role,
            'visible_gpu_name': torch.cuda.get_device_name(0),
            'cuda_device_order': os.environ.get('CUDA_DEVICE_ORDER'),
            'model_id': self.model_cfg['model_id'],
            'model_revision': self.revision(),
            'gpu': self.model_cfg['gpu'],
            'dtype': self.model_cfg['dtype'],
            'thinking_mode': self.model_cfg.get('thinking_mode'),
            'prompt_style': self.prompt_style(),
            'tensor_parallel_size': self.model_cfg['tensor_parallel_size'],
            'engine_args': self.cfg['framework']['vllm_engine_args'],
            'decoding': self.dec,
            'engine_load_sec': self.load_sec,
            'versions': collect_versions(),
        }


# Generation and repair
def main(
    role: str, seed: int, limit: int | None, conditions: list[str] | None, variant: str = 'B5'
) -> dict:
    if variant != 'B5':
        raise ValueError('Only the configured five-claim budget is supported.')
    cfg = settings.llm_config()
    sys_prompt = prompt_templates.SYSTEM_PROMPT
    tag = ''
    for folder in (settings.A_GENERATIONS, settings.LOGS, settings.MANIFESTS):
        folder.mkdir(parents=True, exist_ok=True)
    want_dev = cfg[role]['cuda_visible_devices']
    got_dev = os.environ.get('CUDA_VISIBLE_DEVICES')
    if got_dev != want_dev:
        raise AssertionError(
            f'{role} 은 CUDA_VISIBLE_DEVICES={want_dev} 설정 필요 (현재 {got_dev!r}). '
            f'GPU 배치 불일치 시 두 큐의 충돌 가능성 있음'
        )
    prompts = [
        settings.canonical_names(json.loads(line))
        for line in open(settings.A_PROMPTS / f'prompts_seed{seed}.jsonl')
    ]
    if conditions:
        if not set(conditions).issubset(settings.CONDITIONS):
            raise ValueError('Unknown experiment condition')
        prompts = [p for p in prompts if p['condition'] in set(conditions)]
        if set(conditions) != set(settings.CONDITIONS):
            tag += '_partial'
    if limit is not None and limit <= 0:
        raise ValueError('--limit must be positive')
    if limit:
        keep = sorted({p['case_id'] for p in prompts})[:limit]
        prompts = [p for p in prompts if p['case_id'] in set(keep)]
        tag += '_smoke'
    if not prompts:
        raise AssertionError('생성할 프롬프트가 없음')
    runner = VLLMRunner(role, cfg=cfg)
    info = runner.info()
    info['variant'] = variant
    print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    t0 = time.time()
    raw = runner.generate([(sys_prompt, p['user']) for p in prompts])
    if len(raw) != len(prompts):
        raise RuntimeError('Generation output count does not match the prompt count')
    gen_sec = time.time() - t0
    schema = output.load_schema(cfg)
    out_path = settings.A_GENERATIONS / f'generations_{role}_seed{seed}{tag}.jsonl'
    log_path = settings.LOGS / f'repair_{role}_seed{seed}{tag}.jsonl'
    n_ok = n_repaired = n_failed = 0
    pending: list[tuple[int, str]] = []
    records: list[dict] = []
    with open(log_path, 'w') as lg:
        for i, (p, text) in enumerate(zip(prompts, raw)):
            obj, err = output.try_parse(text, schema, cfg)
            rec = {
                'case_id': p['case_id'],
                'month': p['month'],
                'condition': p['condition'],
                'role': role,
                'model_id': info['model_id'],
                'raw_output': text,
                'parsed': obj,
                'parse_error': err,
                'n_repair_attempts': 0,
            }
            records.append(rec)
            if obj is None:
                pending.append((i, err))
                lg.write(
                    json.dumps(
                        {
                            'stage': 'initial',
                            'case_id': p['case_id'],
                            'condition': p['condition'],
                            'error': err,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
            else:
                n_ok += 1
        for attempt in range(1, cfg['generation']['repair_retries'] + 1):
            if not pending:
                break
            batch = [
                (
                    sys_prompt,
                    prompts[i]['user']
                    + '\n\n'
                    + prompt_templates.repair_prompt(err, records[i]['raw_output'], cfg),
                )
                for i, err in pending
            ]
            fixed = runner.generate(batch)
            if len(fixed) != len(pending):
                raise RuntimeError('Repair output count does not match the requested count')
            still: list[tuple[int, str]] = []
            for (i, _), text in zip(pending, fixed):
                obj, err = output.try_parse(text, schema, cfg)
                records[i]['n_repair_attempts'] = attempt
                records[i]['raw_output_repair'] = text
                if obj is not None:
                    records[i]['parsed'] = obj
                    records[i]['parse_error'] = None
                    n_repaired += 1
                else:
                    records[i]['parse_error'] = err
                    still.append((i, err))
                lg.write(
                    json.dumps(
                        {
                            'stage': f'repair{attempt}',
                            'case_id': records[i]['case_id'],
                            'condition': records[i]['condition'],
                            'ok': obj is not None,
                            'error': err,
                        },
                        ensure_ascii=False,
                    )
                    + '\n'
                )
            pending = still
        n_failed = len(pending)
    temporary = out_path.with_name(f'.{out_path.name}.{uuid.uuid4().hex}')
    try:
        with temporary.open('w', encoding='utf-8') as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + '\n')
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    by_cond: dict[str, dict] = {}
    for r in records:
        d = by_cond.setdefault(r['condition'], {'n': 0, 'parsed': 0, 'repaired': 0})
        d['n'] += 1
        d['parsed'] += int(r['parsed'] is not None)
        d['repaired'] += int(r['n_repair_attempts'] > 0 and r['parsed'] is not None)
    summary = {
        'role': role,
        'seed': seed,
        'variant': variant,
        'budget_B': cfg['generation']['max_factual_sentences'],
        'n_prompts': len(prompts),
        'n_parsed_first_try': n_ok,
        'n_parsed_after_repair': n_repaired,
        'first_pass_rate': round(n_ok / len(prompts), 4),
        'n_unparseable': n_failed,
        'parse_rate': round((n_ok + n_repaired) / len(prompts), 4),
        'per_condition': by_cond,
        'generation_sec': round(gen_sec, 1),
        'throughput_prompts_per_sec': round(len(prompts) / gen_sec, 2),
        'runner': info,
        'generations': str(out_path),
        'repair_log': str(log_path),
    }
    print(
        '\n'
        + json.dumps(
            {k: v for k, v in summary.items() if k != 'runner'}, indent=2, ensure_ascii=False
        ),
        flush=True,
    )
    settings.write_json(settings.MANIFESTS / f'run_llm_{role}_seed{seed}{tag}.json', summary)
    return summary


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument(
        '--role',
        required=True,
        choices=[
            key
            for key, value in settings.llm_config().items()
            if isinstance(value, dict) and 'model_id' in value
        ],
    )
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--limit', type=int, default=None, help='디버그용 사례 수 제한')
    ap.add_argument('--conditions', nargs='*', default=None)
    ap.add_argument('--variant', default='B5', choices=['B5'])
    a = ap.parse_args()
    try:
        main(a.role, a.seed, a.limit, a.conditions, a.variant)
    except IncompatibleStack as e:
        print(f'\n[STOP] {e}', flush=True)
        raise SystemExit(2)
