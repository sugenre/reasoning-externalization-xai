"""Configuration, private paths, and evidence visibility."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / 'config'
INPUTS = (
    Path(os.environ.get('LLM4XAI_INPUT_ROOT', ROOT.parent / 'LLM4XAI_inputs'))
    .expanduser()
    .resolve()
)
OUTPUTS = (
    Path(os.environ.get('LLM4XAI_OUTPUT_ROOT', ROOT.parent / 'LLM4XAI_runs')).expanduser().resolve()
)
ARTIFACTS = OUTPUTS / 'artifacts'
EVALUATION = OUTPUTS / 'evaluation'
LOGS = OUTPUTS / 'logs'
MANIFESTS = OUTPUTS / 'manifests'
A_PROMPTS = ARTIFACTS / 'prompts'
A_GENERATIONS = ARTIFACTS / 'generations'
E_AUTO = EVALUATION / 'automatic'
E_STATS = EVALUATION / 'statistics'
_CFG_CACHE: dict[str, dict] = {}
FEATURE_ALIASES = {'high_52w_ratio': 'high_12m_ratio'}


def canonical_names(value: Any) -> Any:
    """Feature names in input records and evidence identifiers."""
    if isinstance(value, str):
        for source, target in FEATURE_ALIASES.items():
            value = re.sub(r'\b' + re.escape(source) + r'\b', target, value)
        return value
    if isinstance(value, dict):
        return {canonical_names(k): canonical_names(v) for k, v in value.items()}
    if isinstance(value, list):
        return [canonical_names(v) for v in value]
    if isinstance(value, tuple):
        return tuple(canonical_names(v) for v in value)
    return value


def load_yaml(name: str) -> dict:
    if name not in _CFG_CACHE:
        with (CONFIG_DIR / name).open(encoding='utf-8') as handle:
            _CFG_CACHE[name] = yaml.safe_load(handle)
    return _CFG_CACHE[name]


def frozen_config() -> dict:
    return load_yaml('settings.yaml')['experiment']


def feature_metadata() -> dict:
    return load_yaml('feature_metadata.yaml')


def llm_config() -> dict:
    return load_yaml('settings.yaml')['llm']


def input_path(key: str, seed: int | None = None) -> Path:
    template = frozen_config()['paths'][key]
    path = INPUTS / (template.format(seed=seed) if seed is not None else template)
    if not path.is_file():
        raise FileNotFoundError(f'Missing input: {path}. Set LLM4XAI_INPUT_ROOT.')
    return path


def database_path(seed: int = 42) -> Path:
    override = os.environ.get('LLM4XAI_DB_PATH')
    return (
        Path(override).expanduser().resolve()
        if override
        else ARTIFACTS / f'evidence_seed{seed}.duckdb'
    )


def write_json(path: Path, obj: object, indent: int = 2) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, indent=indent, ensure_ascii=False, default=_default)
    return path


def _default(value: object) -> object:
    import numpy as np

    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f'Cannot serialize {type(value)}')


def rel(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


C0_EVIDENCE_CONTROL = 'C0-Evidence Control'
C1_RAW = 'C1-Raw'
C2_NUMERICAL = 'C2-Numerical'
C3_RELATIONAL = 'C3-Relational'
C3_ANALOG = 'C3-Analog'
CONDITIONS = [C0_EVIDENCE_CONTROL, C1_RAW, C2_NUMERICAL, C3_RELATIONAL, C3_ANALOG]
VISIBILITY = {
    'base': CONDITIONS,
    'raw': CONDITIONS[1:],
    'numerical': CONDITIONS[2:],
    'relational': CONDITIONS[3:],
    'analog': CONDITIONS[4:],
}
