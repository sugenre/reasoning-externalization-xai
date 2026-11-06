"""Prediction/SHAP inputs and train-fitted preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
from pathlib import Path
import settings

_FEATURE_CONFIG = settings.frozen_config()['features']
MODEL_ORDER: list[str] = list(_FEATURE_CONFIG['model_order'])
RETRIEVAL_ORDER: list[str] = list(_FEATURE_CONFIG['retrieval_order'])
NF: int = int(_FEATURE_CONFIG['n'])
REMOVED: list[str] = list(_FEATURE_CONFIG['removed'])
MODEL_TO_RETRIEVAL: np.ndarray = np.array(
    [MODEL_ORDER.index(f) for f in RETRIEVAL_ORDER], dtype=np.int64
)
RETRIEVAL_TO_MODEL: np.ndarray = np.array(
    [RETRIEVAL_ORDER.index(f) for f in MODEL_ORDER], dtype=np.int64
)
P_COLS_RETRIEVAL: list[str] = [f'P_{f}' for f in RETRIEVAL_ORDER]
PHI_COLS_RETRIEVAL: list[str] = [f'phi_{f}' for f in RETRIEVAL_ORDER]


# Feature contract
def check_orders() -> None:
    """Feature count, order, and metadata checks."""
    if len(MODEL_ORDER) != NF or len(RETRIEVAL_ORDER) != NF:
        raise AssertionError(
            f'피처 개수가 {NF} 가 아님: model={len(MODEL_ORDER)} retrieval={len(RETRIEVAL_ORDER)}'
        )
    if set(MODEL_ORDER) != set(RETRIEVAL_ORDER):
        raise AssertionError('model_order 와 retrieval_order 의 피처 집합이 다름')
    for bad in REMOVED:
        if bad in MODEL_ORDER or bad in RETRIEVAL_ORDER:
            raise AssertionError(f'제거되어야 할 피처가 남아 있음: {bad}')
    if len(set(MODEL_ORDER)) != NF:
        raise AssertionError('model_order 에 중복 피처가 있음')
    meta = set(settings.feature_metadata()['features'])
    if meta != set(MODEL_ORDER):
        raise AssertionError(
            f'feature_metadata.yaml 이 30피처를 정확히 포함하지 않음: '
            f'빠짐={set(MODEL_ORDER) - meta} 잉여={meta - set(MODEL_ORDER)}'
        )


def assert_matches_booster(booster_feats: list[str]) -> None:
    """Frozen predictor feature-order validation."""
    booster_feats = settings.canonical_names(booster_feats)
    if list(booster_feats) != MODEL_ORDER:
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(booster_feats, MODEL_ORDER)) if a != b]
        raise AssertionError(f'booster 피처 순서가 MODEL_ORDER 와 다름: {diff[:5]}')


def to_retrieval_axis(arr_model_axis: np.ndarray) -> np.ndarray:
    """Last-axis conversion from MODEL_ORDER to RETRIEVAL_ORDER."""
    return np.take(arr_model_axis, MODEL_TO_RETRIEVAL, axis=-1)


def assert_permutation_invariance(
    rng: np.random.Generator | None = None, n_trials: int = 200, tol: float = 1e-12
) -> float:
    """Euclidean-distance invariance under a shared feature permutation."""
    rng = rng or np.random.default_rng(0)
    worst = 0.0
    for _ in range(n_trials):
        a = rng.normal(size=(6, NF))
        b = rng.normal(size=(6, NF))
        d_model = float(np.linalg.norm((a - b).ravel()))
        d_retr = float(np.linalg.norm((to_retrieval_axis(a) - to_retrieval_axis(b)).ravel()))
        worst = max(worst, abs(d_model - d_retr))
    if worst > tol:
        raise AssertionError(f'치환 불변성 위반: 최대 오차 {worst:.3e} > {tol:g}')
    return worst


def describe_feature(feature: str) -> dict:
    return settings.feature_metadata()['features'][feature]


_DATA_CONFIG = settings.frozen_config()
FEATS = list(_DATA_CONFIG['features']['model_order'])
_INPUT_FEATURE_COUNT = len(FEATS)
DROP = list(_DATA_CONFIG['features']['removed'])
TGT = _DATA_CONFIG['split']['target']
TR_LO, TR_HI = (_DATA_CONFIG['split']['train']['lo'], _DATA_CONFIG['split']['train']['hi'])
VA_LO, VA_HI = (
    _DATA_CONFIG['split']['validation']['lo'],
    _DATA_CONFIG['split']['validation']['hi'],
)
TE_LO, TE_HI = (_DATA_CONFIG['split']['test']['lo'], _DATA_CONFIG['split']['test']['hi'])
WIN_LO, WIN_HI = (0.01, 0.99)


# Panel preprocessing
def read_feature_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Canonical feature columns with source-file compatibility."""
    names = pq.read_schema(path).names
    aliases = {
        prefix + source: prefix + target
        for source, target in settings.FEATURE_ALIASES.items()
        for prefix in ('', 'phi_', 'A_', 'P_')
        if prefix + source in names
    }
    if any(target in names for target in aliases.values()):
        raise ValueError('입력 파일에 같은 피처의 두 이름이 함께 있음')
    source_names = {target: source for source, target in aliases.items()}
    selected = None if columns is None else [source_names.get(c, c) for c in columns]
    return pd.read_parquet(path, columns=selected).rename(columns=aliases)


class Data:
    """Train-fitted panel: features F (n, 30), targets y (n,), and split indices."""

    def __init__(self, winsorize: bool = True, verbose: bool = True) -> None:
        df = read_feature_parquet(
            settings.input_path('panel_parquet'), columns=['PERMNO', 'date', TGT] + FEATS
        )
        df['date'] = pd.to_datetime(df['date'])
        df = df.dropna(subset=[TGT])
        df[FEATS] = df[FEATS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df = df.sort_values(['PERMNO', 'date']).reset_index(drop=True)
        F64 = df[FEATS].to_numpy(np.float64)
        self.y = df[TGT].to_numpy(np.float32)
        self.dates = df['date'].values
        self.permno = df['PERMNO'].to_numpy()
        d = df['date']

        def sel(lo, hi):
            m = (d >= pd.Timestamp(lo)).values & (d <= pd.Timestamp(hi)).values
            return np.flatnonzero(m)

        self.pos_tr = sel(TR_LO, TR_HI)
        self.pos_va = sel(VA_LO, VA_HI)
        self.pos_te = sel(TE_LO, TE_HI)
        self.clip_lo = self.clip_hi = None
        n_clip = 0
        if winsorize:
            lo = np.quantile(F64[self.pos_tr], WIN_LO, axis=0)
            hi = np.quantile(F64[self.pos_tr], WIN_HI, axis=0)
            n_clip = int((F64 < lo).sum() + (F64 > hi).sum())
            self.clip_lo, self.clip_hi = (lo, hi)
            self.was_clipped = (F64 < lo) | (F64 > hi)
            F64 = np.clip(F64, lo, hi)
        else:
            self.was_clipped = np.zeros(F64.shape, bool)
        mu = F64[self.pos_tr].mean(0)
        sd = F64[self.pos_tr].std(0)
        sd[sd < 1e-08] = 1.0
        self.mu, self.sd = (mu, sd)
        self.F = ((F64 - mu) / sd).astype(np.float32)
        del F64
        if verbose:
            print(
                f'[Data] tr {TR_LO[:7]}~{TR_HI[:7]} / va {VA_LO[:7]}~{VA_HI[:7]} / '
                f'te {TE_LO[:7]}~{TE_HI[:7]}',
                flush=True,
            )
            print(
                f'[Data] 피처 {_INPUT_FEATURE_COUNT}개 (제외 {DROP})  L 필터 없음  '
                f'윈저화 {WIN_LO:.0%}/{WIN_HI:.0%}  통계=float64',
                flush=True,
            )
            print(f'       클리핑 {n_clip:,}개 ({100 * n_clip / self.F.size:.2f}%)', flush=True)
            print(
                f'       train {len(self.pos_tr):,} / val {len(self.pos_va):,} / '
                f'test {len(self.pos_te):,}',
                flush=True,
            )

    def X(self, pos):
        return self.F[pos]

    def target(self, pos):
        return self.y[pos]


TEST_LO, TEST_HI = ('2012-01', '2023-11')


# SHAP inputs
@dataclass
class GlobalSHAP:
    """Monthly global SHAP arrays in RETRIEVAL_ORDER."""

    months: np.ndarray
    A: np.ndarray
    P: np.ndarray
    split: np.ndarray
    n_stocks: np.ndarray

    def index_of(self, month: str) -> int:
        i = int(np.searchsorted(self.months, month))
        if i >= len(self.months) or self.months[i] != month:
            raise KeyError(f'월 {month} 이 monthly_state 에 없음')
        return i

    def rank(self, month: str) -> dict[str, int]:
        """Descending P ranks; stable ties follow RETRIEVAL_ORDER."""
        row = self.P[self.index_of(month)]
        order = np.argsort(-row, kind='stable')
        out: dict[str, int] = {}
        for r, j in enumerate(order, start=1):
            out[RETRIEVAL_ORDER[j]] = r
        return out

    def test_months(self) -> list[str]:
        return [m for m, s in zip(self.months, self.split) if s == 'test']


def load_global_shap(seed: int, validate: bool = True) -> GlobalSHAP:
    df = read_feature_parquet(settings.input_path('monthly_state', seed))
    df = df.sort_values('month').reset_index(drop=True)
    missing = [c for c in P_COLS_RETRIEVAL if c not in df.columns]
    if missing:
        raise AssertionError(f'monthly_state 에 P 열이 없음: {missing[:5]}')
    P = df[P_COLS_RETRIEVAL].to_numpy(np.float64)
    A = df[[f'A_{f}' for f in RETRIEVAL_ORDER]].to_numpy(np.float64)
    g = GlobalSHAP(
        months=df['month'].to_numpy(),
        A=A,
        P=P,
        split=df['split'].to_numpy(),
        n_stocks=df['n_stocks'].to_numpy(np.int64),
    )
    if validate:
        validate_global(g)
    return g


def validate_global(g: GlobalSHAP, tol: float = 1e-09) -> dict:
    """Nonnegative importance, unit row sums, and A-to-P consistency."""
    s = g.P.sum(axis=1)
    err = float(np.abs(s - 1.0).max())
    if err > tol:
        raise AssertionError(f'Σ_j P[j,t] ≠ 1 (최대 편차 {err:.3e} > {tol:g})')
    if (g.P < -tol).any():
        raise AssertionError('P 에 음수가 있음 — mean|φ| 기반이므로 불가능')
    recomputed = g.A / g.A.sum(axis=1, keepdims=True)
    consist = float(np.abs(recomputed - g.P).max())
    if consist > 1e-06:
        raise AssertionError(f'A 로 재계산한 P 가 저장된 P 와 다름 (최대 {consist:.3e})')
    if not np.all(g.months[:-1] < g.months[1:]):
        raise AssertionError('monthly_state 의 월이 오름차순·유일성 조건 위반')
    return {
        'max_abs_sumP_minus_1': err,
        'max_abs_A_to_P_inconsistency': consist,
        'n_months': int(len(g.months)),
    }


def load_local_shap(
    seed: int, months: list[str] | None = None, columns: list[str] | None = None
) -> pd.DataFrame:
    """Local SHAP columns with PERMNO, month, prediction, and target."""
    base = ['PERMNO', 'month', 'pred', 'y']
    cols = None if columns is None else base + [c for c in columns if c not in base]
    df = read_feature_parquet(settings.input_path('shap_local', seed), columns=cols)
    if months is not None:
        df = df[df['month'].isin(set(months))]
    return df.reset_index(drop=True)


def validate_local(
    seed: int, g: GlobalSHAP, expected_rows: int, recompute_tol: float = 1e-06
) -> dict:
    """Local keys, feature coverage, and recomputed monthly mean absolute SHAP."""
    df = load_local_shap(seed)
    report: dict = {}
    report['n_rows'] = int(len(df))
    if len(df) != expected_rows:
        raise AssertionError(f'local SHAP 행수 {len(df):,} ≠ 기대 {expected_rows:,}')
    missing = [c for c in PHI_COLS_RETRIEVAL if c not in df.columns]
    if missing:
        raise AssertionError(f'local SHAP 에 없는 피처 열: {missing}')
    extra_phi = [c for c in df.columns if c.startswith('phi_') and c not in PHI_COLS_RETRIEVAL]
    if extra_phi:
        raise AssertionError(f'local SHAP 에 허용되지 않은 phi 열: {extra_phi}')
    report['n_phi_cols'] = len(PHI_COLS_RETRIEVAL)
    dup = int(df.duplicated(['PERMNO', 'month']).sum())
    if dup:
        raise AssertionError(f'stock-month 키 중복 {dup}건')
    report['n_duplicate_keys'] = dup
    months = sorted(df['month'].unique())
    report['n_months'] = len(months)
    report['month_min'], report['month_max'] = (months[0], months[-1])
    if months[0] != TEST_LO or months[-1] != TEST_HI:
        raise AssertionError(f'local SHAP 월 범위 {months[0]}~{months[-1]} ≠ {TEST_LO}~{TEST_HI}')
    absphi = df[PHI_COLS_RETRIEVAL].astype(np.float64).abs()
    absphi['month'] = df['month']
    A_re = absphi.groupby('month').mean()[PHI_COLS_RETRIEVAL].to_numpy(np.float64)
    idx = [g.index_of(m) for m in months]
    A_ref = g.A[idx]
    denom = np.maximum(np.abs(A_ref), 1e-12)
    rel_err = float((np.abs(A_re - A_ref) / denom).max())
    report['max_rel_err_A_recomputed'] = rel_err
    if rel_err > 0.0001:
        raise AssertionError(
            f'local φ 로 재계산한 A 가 monthly_state 와 다름 (상대오차 {rel_err:.3e})'
        )
    P_re = A_re / A_re.sum(1, keepdims=True)
    report['max_abs_err_P_recomputed'] = float(np.abs(P_re - g.P[idx]).max())
    report['recompute_needed'] = False
    return report


# Case feature values
def extract_model_inputs(cases: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Case-level standardized, raw, and percentile feature values."""
    if list(FEATS) != MODEL_ORDER:
        raise AssertionError('preprocessing.FEATS 가 MODEL_ORDER 와 다름 — 계약 위반')
    d = Data(verbose=verbose)
    month_key = pd.Series(pd.to_datetime(d.dates)).dt.to_period('M').astype(str).to_numpy()
    key = pd.DataFrame({'PERMNO': d.permno, 'month': month_key})
    key['row'] = np.arange(len(key), dtype=np.int64)
    want = cases[['case_id', 'PERMNO', 'month']].copy()
    want['PERMNO'] = want['PERMNO'].astype(key['PERMNO'].dtype)
    merged = want.merge(key, on=['PERMNO', 'month'], how='left', validate='one_to_one')
    if merged['row'].isna().any():
        bad = merged[merged['row'].isna()][['case_id']].head()
        raise AssertionError(f"패널에서 찾을 수 없는 사례가 있음: {bad.to_dict('records')}")
    rows = merged['row'].to_numpy(np.int64)
    Z_model = d.F[rows]
    Z = to_retrieval_axis(Z_model.astype(np.float64))
    panel = read_feature_parquet(
        settings.input_path('panel_parquet'), columns=['PERMNO', 'date'] + RETRIEVAL_ORDER
    )
    panel['month'] = pd.to_datetime(panel['date']).dt.to_period('M').astype(str)
    months = sorted(cases['month'].unique())
    panel = panel[panel['month'].isin(set(months))]
    panel = panel.replace([np.inf, -np.inf], np.nan).fillna({c: 0.0 for c in RETRIEVAL_ORDER})
    idx = {(int(p), m): i for i, (p, m) in enumerate(zip(cases['PERMNO'], cases['month']))}
    raw = np.full((len(cases), NF), np.nan)
    pct = np.full((len(cases), NF), np.nan)
    for m, blk in panel.groupby('month', sort=True):
        vals = blk[RETRIEVAL_ORDER].to_numpy(np.float64)
        n = len(blk)
        ranks = np.argsort(np.argsort(vals, axis=0, kind='stable'), axis=0)
        pctile = np.floor(100.0 * ranks / n)
        for local_i, p in enumerate(blk['PERMNO'].to_numpy()):
            k = idx.get((int(p), m))
            if k is not None:
                raw[k] = vals[local_i]
                pct[k] = pctile[local_i]
    if np.isnan(raw).any():
        raise AssertionError('원단위 값이 채워지지 않은 사례가 있음')
    out = pd.DataFrame(
        {
            'case_id': cases['case_id'].to_numpy(),
            'PERMNO': cases['PERMNO'].to_numpy(),
            'month': cases['month'].to_numpy(),
        }
    )
    for j, f in enumerate(RETRIEVAL_ORDER):
        out[f'z_{f}'] = Z[:, j]
        out[f'raw_{f}'] = raw[:, j]
        out[f'pct_{f}'] = pct[:, j].astype(np.int64)
    return out
