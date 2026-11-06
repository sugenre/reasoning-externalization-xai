"""Temporal evidence and historical analog retrieval."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
import data
import settings

L = 6
K = 5
NMS_M = 6
ELIGIBILITY_GAP = 7
Q_SHAPE = (L, data.NF)
Q_FLAT_DIM = L * data.NF


def month_to_int(m: str) -> int:
    """Calendar-month index from YYYY-MM."""
    y, mm = m.split('-')
    return int(y) * 12 + int(mm) - 1


def int_to_month(k: int) -> str:
    return f'{k // 12:04d}-{k % 12 + 1:02d}'


@dataclass(frozen=True)
class Analog:
    analog_rank: int
    analog_start: str
    analog_end: str
    endpoint_s: str
    euclidean_distance: float
    subsequent_market_return: float
    subsequent_return_month: str

    def to_dict(self) -> dict:
        return asdict(self)


# Historical analogs
class TrajectoryLibrary:
    """Monthly SHAP-importance change trajectories."""

    def __init__(self, g: data.GlobalSHAP, market_return: dict[str, float]):
        self.g = g
        self.market_return = market_return
        self.months = list(g.months)
        self.midx = {m: i for i, m in enumerate(self.months)}
        self.mint = np.array([month_to_int(m) for m in self.months], dtype=np.int64)
        M, NF = g.P.shape
        self.dP = np.full((M, NF), np.nan, dtype=np.float64)
        for i in range(1, M):
            if self.mint[i] - self.mint[i - 1] == 1:
                self.dP[i] = g.P[i] - g.P[i - 1]
        self.test_mask = g.split == 'test'
        self._Q_cache: dict[str, np.ndarray] = {}
        self.library_endpoints = self._build_library()
        self.library_int = np.array(
            [month_to_int(s) for s in self.library_endpoints], dtype=np.int64
        )
        self.library_Q = (
            np.stack([self.Q(s) for s in self.library_endpoints])
            if self.library_endpoints
            else np.zeros((0, Q_FLAT_DIM))
        )

    def support_months(self, endpoint: str) -> list[str]:
        """Seven P observations supporting six monthly changes."""
        e = month_to_int(endpoint)
        return [int_to_month(e - k) for k in range(L, -1, -1)]

    def window_months(self, endpoint: str) -> list[str]:
        """Six change-vector months ending at the selected endpoint."""
        e = month_to_int(endpoint)
        return [int_to_month(e - k) for k in range(L - 1, -1, -1)]

    def has_valid_Q(self, endpoint: str) -> bool:
        sup = self.support_months(endpoint)
        if any((m not in self.midx for m in sup)):
            return False
        rows = [self.midx[m] for m in self.window_months(endpoint)]
        return not np.isnan(self.dP[rows]).any()

    def Q(self, endpoint: str) -> np.ndarray:
        """Flattened (6, 30) change trajectory in RETRIEVAL_ORDER."""
        if endpoint in self._Q_cache:
            return self._Q_cache[endpoint]
        if not self.has_valid_Q(endpoint):
            raise ValueError(f'Q[{endpoint}] 가 유효하지 않음')
        rows = [self.midx[m] for m in self.window_months(endpoint)]
        Q = self.dP[rows]
        if Q.shape != Q_SHAPE:
            raise AssertionError(f'Q shape {Q.shape} ≠ {Q_SHAPE}')
        v = Q.ravel()
        if v.shape[0] != Q_FLAT_DIM:
            raise AssertionError(f'flatten 차원 {v.shape[0]} ≠ {Q_FLAT_DIM}')
        self._Q_cache[endpoint] = v
        return v

    def Q_matrix(self, endpoint: str) -> np.ndarray:
        return self.Q(endpoint).reshape(Q_SHAPE)

    def _build_library(self) -> list[str]:
        """Test-only seven-month supports with observed subsequent returns."""
        test_set = {m for m, ok in zip(self.months, self.test_mask) if ok}
        out = []
        for m in self.months:
            if m not in test_set:
                continue
            sup = self.support_months(m)
            if not all((x in test_set for x in sup)):
                continue
            if not self.has_valid_Q(m):
                continue
            nxt = int_to_month(month_to_int(m) + 1)
            if nxt not in self.market_return:
                continue
            out.append(m)
        return out

    def eligible(self, t: str) -> np.ndarray:
        """Historical endpoints at least seven months before the query."""
        cutoff = month_to_int(t) - ELIGIBILITY_GAP
        return np.flatnonzero(self.library_int <= cutoff)

    def retrieve(self, t: str, k: int = K, m_nms: int = NMS_M) -> list[Analog]:
        """Distance-ranked analogs with temporal spacing; at most k results."""
        q = self.Q(t)
        cand = self.eligible(t)
        if cand.size == 0:
            return []
        d = np.linalg.norm(self.library_Q[cand] - q[None, :], axis=1)
        order = np.lexsort((self.library_int[cand], d))
        chosen: list[int] = []
        chosen_int: list[int] = []
        for oi in order:
            gi = int(cand[oi])
            si = int(self.library_int[gi])
            if all((abs(si - c) >= m_nms for c in chosen_int)):
                chosen.append(gi)
                chosen_int.append(si)
                if len(chosen) == k:
                    break
        out = []
        for rank, gi in enumerate(chosen, start=1):
            s = self.library_endpoints[gi]
            nxt = int_to_month(month_to_int(s) + 1)
            win = self.window_months(s)
            dist = float(np.linalg.norm(self.library_Q[gi] - q))
            out.append(
                Analog(
                    analog_rank=rank,
                    analog_start=win[0],
                    analog_end=win[-1],
                    endpoint_s=s,
                    euclidean_distance=dist,
                    subsequent_market_return=float(self.market_return[nxt]),
                    subsequent_return_month=nxt,
                )
            )
        return out

    def n_survivors(self, t: str, m_nms: int = NMS_M) -> int:
        """Full candidate count after temporal spacing, before the K limit."""
        return len(self.retrieve(t, k=10**9, m_nms=m_nms))


def analog_summary(analogs: list[Analog]) -> dict:
    """Precomputed historical return counts and summaries."""
    r = np.array([a.subsequent_market_return for a in analogs], dtype=np.float64)
    return {
        'positive_count': int((r > 0).sum()),
        'negative_count': int((r < 0).sum()),
        'zero_count': int((r == 0).sum()),
        'mean_subsequent_return': float(r.mean()) if r.size else float('nan'),
        'median_subsequent_return': float(np.median(r)) if r.size else float('nan'),
        'n_analogs': int(r.size),
    }


def check_analog_constraints(t: str, analogs: list[Analog], k: int = K, m_nms: int = NMS_M) -> None:
    """Eligibility, temporal spacing, analog count, and outcome alignment."""
    if len(analogs) != k:
        raise AssertionError(f'{t}: analog 개수 {len(analogs)} ≠ K={k}')
    ti = month_to_int(t)
    ints = [month_to_int(a.endpoint_s) for a in analogs]
    for a, si in zip(analogs, ints):
        if si > ti - ELIGIBILITY_GAP:
            raise AssertionError(f'{t}: 후보 s={a.endpoint_s} 가 s ≤ t−{ELIGIBILITY_GAP} 위반')
        if month_to_int(a.subsequent_return_month) != si + 1:
            raise AssertionError(
                f'{t}: r[s+1] 정렬 오류 ({a.endpoint_s} → {a.subsequent_return_month})'
            )
        if month_to_int(a.analog_end) != si:
            raise AssertionError(f'{t}: analog_end 가 endpoint 와 다름')
        if month_to_int(a.analog_start) != si - (L - 1):
            raise AssertionError(f'{t}: analog_start 가 L={L} 창과 맞지 않음')
    for i in range(len(ints)):
        for j in range(i + 1, len(ints)):
            if abs(ints[i] - ints[j]) < m_nms:
                raise AssertionError(
                    f'{t}: temporal NMS m={m_nms} 위반 '
                    f'({analogs[i].endpoint_s}, {analogs[j].endpoint_s})'
                )
    dists = [a.euclidean_distance for a in analogs]
    if any((dists[i] > dists[i + 1] + 1e-12 for i in range(len(dists) - 1))):
        raise AssertionError(f'{t}: analog_rank 가 거리 오름차순이 아님')


_X = np.arange(L, dtype=np.float64)
_XC = _X - _X.mean()
_DEN = float((_XC**2).sum())


# Temporal descriptors
@dataclass(frozen=True)
class FeatureTemporalEvidence:
    feature: str
    current_importance: float
    window_change_delta6P: float
    trend_slope: float
    trend_direction: str
    rank_start: int
    rank_end: int
    rank_change: int
    raw_sequence: list[float]
    sequence_months: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def window_months(t: str) -> list[str]:
    ti = month_to_int(t)
    return [int_to_month(ti - k) for k in range(L - 1, -1, -1)]


def ols_slope(y: np.ndarray) -> float:
    """OLS slope on equally spaced month indices 0 through L - 1."""
    yc = y - y.mean()
    return float((_XC * yc).sum() / _DEN)


def direction(slope: float, eps: float = 0.0) -> str:
    if slope > eps:
        return 'increasing'
    if slope < -eps:
        return 'decreasing'
    return 'flat'


def compute(g: data.GlobalSHAP, t: str, feats: list[str]) -> dict[str, FeatureTemporalEvidence]:
    wm = window_months(t)
    for m in wm:
        g.index_of(m)
    rows = [g.index_of(m) for m in wm]
    P_win = g.P[rows]
    rank_start = g.rank(wm[0])
    rank_end = g.rank(wm[-1])
    out: dict[str, FeatureTemporalEvidence] = {}
    for f in feats:
        j = data.RETRIEVAL_ORDER.index(f)
        seq = P_win[:, j]
        slope = ols_slope(seq)
        out[f] = FeatureTemporalEvidence(
            feature=f,
            current_importance=float(seq[-1]),
            window_change_delta6P=float(seq[-1] - seq[0]),
            trend_slope=slope,
            trend_direction=direction(slope),
            rank_start=int(rank_start[f]),
            rank_end=int(rank_end[f]),
            rank_change=int(rank_start[f] - rank_end[f]),
            raw_sequence=[float(v) for v in seq],
            sequence_months=list(wm),
        )
    return out


# Universe returns
def build_market_return(seed: int) -> pd.DataFrame:
    """Subsequent equal-weight returns over the evaluation universe."""
    df = pd.read_parquet(settings.input_path('shap_local', seed), columns=['month', 'y'])
    g = df.groupby('month')['y']
    out = pd.DataFrame(
        {
            'month_s': g.mean().index.to_numpy(),
            'market_return_equal_weighted': g.mean().to_numpy(np.float64),
            'n_stocks': g.size().to_numpy(np.int64),
        }
    )
    out['subsequent_month'] = [int_to_month(month_to_int(m) + 1) for m in out['month_s']]
    return out[['month_s', 'subsequent_month', 'n_stocks', 'market_return_equal_weighted']]


def as_lookup(df: pd.DataFrame) -> dict[str, float]:
    """Universe-return lookup keyed by realization month."""
    return dict(zip(df['subsequent_month'], df['market_return_equal_weighted']))
