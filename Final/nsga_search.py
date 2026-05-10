"""Phase 8 NSGA-II: 12차원 σ → Pareto front (max W, min σ_norm, min RA).

12차원 σ (B-결과 3 재설계):
    타자 4 (team-wide multiplier):
        hr_mult, bb_mult, k_mult, single_mult
    투수 6 (tier multiplier):
        closer.K_pct, closer.BB_pct, closer.BABIP, closer.HR_pct,
        setup.K_pct, starter.HR_pct
    공통 2 (모든 tier 일괄 multiplier):
        team_K_pct_mult, team_BB_pct_mult

목적함수 (3-objective, 모두 minimize 형태로 변환):
    - max W       → -W
    - min σ_norm  → RMS(|x - 1|)
    - min RA

Usage:
    from nsga_search import run_nsga
    res = run_nsga(pop_size=50, n_gen=15, n_seasons=20)
    # res.X: (n_pareto, 12), res.F: (n_pareto, 3) — F 컬럼 [-W, σ_norm, RA]
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from integrated_sim import run_integrated_simulation, _ensure_loaded


# ──────────────────────────────────────────────────
# σ 12차원 정의
# ──────────────────────────────────────────────────

SIGMA_DIMS = [
    # 타자 (team-wide multiplier)
    {'id': 'h_hr',     'group': 'hitter',  'sub': 'team',    'key': 'hr_mult',     'lo': 0.85, 'hi': 1.30},
    {'id': 'h_bb',     'group': 'hitter',  'sub': 'team',    'key': 'bb_mult',     'lo': 0.85, 'hi': 1.30},
    {'id': 'h_k',      'group': 'hitter',  'sub': 'team',    'key': 'k_mult',      'lo': 0.70, 'hi': 1.15},
    {'id': 'h_single', 'group': 'hitter',  'sub': 'team',    'key': 'single_mult', 'lo': 0.85, 'hi': 1.30},
    # 투수 tier
    {'id': 'p_cl_K',   'group': 'pitcher', 'sub': 'closer',  'key': 'K_pct',  'lo': 0.80, 'hi': 1.25},
    {'id': 'p_cl_BB',  'group': 'pitcher', 'sub': 'closer',  'key': 'BB_pct', 'lo': 0.75, 'hi': 1.20},
    {'id': 'p_cl_BAB', 'group': 'pitcher', 'sub': 'closer',  'key': 'BABIP',  'lo': 0.80, 'hi': 1.20},
    {'id': 'p_cl_HR',  'group': 'pitcher', 'sub': 'closer',  'key': 'HR_pct', 'lo': 0.50, 'hi': 1.30},
    {'id': 'p_su_K',   'group': 'pitcher', 'sub': 'setup',   'key': 'K_pct',  'lo': 0.80, 'hi': 1.25},
    {'id': 'p_st_HR',  'group': 'pitcher', 'sub': 'starter', 'key': 'HR_pct', 'lo': 0.50, 'hi': 1.30},
    # 공통 (pitcher.common)
    {'id': 'c_K',      'group': 'pitcher', 'sub': 'common',  'key': 'team_K_pct_mult',  'lo': 0.90, 'hi': 1.15},
    {'id': 'c_BB',     'group': 'pitcher', 'sub': 'common',  'key': 'team_BB_pct_mult', 'lo': 0.90, 'hi': 1.15},
]
N_VAR = len(SIGMA_DIMS)
LOWER = np.array([d['lo'] for d in SIGMA_DIMS])
UPPER = np.array([d['hi'] for d in SIGMA_DIMS])
DIM_IDS = [d['id'] for d in SIGMA_DIMS]


def vector_to_adjustments(x):
    """12차원 σ 벡터 → (hitter_adjustments, pitcher_adjustments) tuple."""
    hitter: dict = {}
    pitcher: dict = {}
    for i, dim in enumerate(SIGMA_DIMS):
        target = hitter if dim['group'] == 'hitter' else pitcher
        target.setdefault(dim['sub'], {})[dim['key']] = float(x[i])
    return hitter, pitcher


def sigma_norm(x) -> float:
    """RMS(|x - 1|) — multiplier가 1에서 떨어진 정도 (정책 비용 proxy)."""
    return float(np.sqrt(np.mean((np.asarray(x, dtype=float) - 1.0) ** 2)))


def evaluate_sigma(x, n_seasons: int = 20, seed_offset: int = 0) -> dict:
    """단일 σ 벡터 평가 → {W, RS, RA, sigma_norm}."""
    h, p = vector_to_adjustments(x)
    seeds = range(seed_offset, seed_offset + n_seasons)
    df, _ = run_integrated_simulation(
        n_seasons=n_seasons,
        hitter_adjustments=h,
        pitcher_adjustments=p,
        seeds=seeds,
    )
    return {
        'W':           float(df['W'].mean()),
        'W_std':       float(df['W'].std()),
        'RS':          float(df['RS'].mean()),
        'RA':          float(df['RA'].mean()),
        'sigma_norm':  sigma_norm(x),
    }


# ──────────────────────────────────────────────────
# pymoo 통합
# ──────────────────────────────────────────────────

from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization import StarmapParallelization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS


class TexSigmaProblem(ElementwiseProblem):
    """3-objective σ 최적화 — F = [-W, σ_norm, RA]."""

    def __init__(self, n_seasons: int = 20, seed_offset: int = 0, **kw):
        super().__init__(n_var=N_VAR, n_obj=3, xl=LOWER, xu=UPPER, **kw)
        self.n_seasons = n_seasons
        self.seed_offset = seed_offset

    def _evaluate(self, x, out, *args, **kwargs):
        ev = evaluate_sigma(x, n_seasons=self.n_seasons, seed_offset=self.seed_offset)
        out['F'] = [-ev['W'], ev['sigma_norm'], ev['RA']]


def _build_algorithm(pop_size: int) -> NSGA2:
    return NSGA2(
        pop_size=pop_size,
        sampling=LHS(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )


def _result_to_df(res) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X = res.X if res.X is not None else np.empty((0, N_VAR))
    F = res.F if res.F is not None else np.empty((0, 3))
    cols = {'W': -F[:, 0], 'sigma_norm': F[:, 1], 'RA': F[:, 2]}
    for i, did in enumerate(DIM_IDS):
        cols[did] = X[:, i]
    return X, F, pd.DataFrame(cols)


def run_nsga(
    pop_size: int = 50,
    n_gen: int = 15,
    n_seasons: int = 20,
    seed: int = 42,
    seed_offset: int = 0,
    verbose: bool = True,
) -> dict:
    """NSGA-II 단일 프로세스 실행."""
    _ensure_loaded()
    problem = TexSigmaProblem(n_seasons=n_seasons, seed_offset=seed_offset)
    algorithm = _build_algorithm(pop_size)
    termination = get_termination('n_gen', n_gen)
    res = minimize(problem, algorithm, termination, seed=seed,
                   verbose=verbose, save_history=False)
    X, F, df = _result_to_df(res)
    return {'res': res, 'X': X, 'F': F, 'df': df}


def run_nsga_parallel(
    pop_size: int = 50,
    n_gen: int = 15,
    n_seasons: int = 20,
    seed: int = 42,
    seed_offset: int = 0,
    n_proc: int = 6,
    verbose: bool = True,
) -> dict:
    """NSGA-II 멀티프로세스 실행 (macOS spawn 안전).

    Worker 첫 호출 시 _ensure_loaded()가 simulator bundle 캐시 디스크에서 로드(~5초).
    이후 동일 worker 재사용 시 module-level 캐시.
    """
    import multiprocessing as mp

    ctx = mp.get_context('spawn')
    with ctx.Pool(n_proc) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = TexSigmaProblem(
            n_seasons=n_seasons, seed_offset=seed_offset,
            elementwise_runner=runner,
        )
        algorithm = _build_algorithm(pop_size)
        termination = get_termination('n_gen', n_gen)
        res = minimize(problem, algorithm, termination, seed=seed,
                       verbose=verbose, save_history=False)
    X, F, df = _result_to_df(res)
    return {'res': res, 'X': X, 'F': F, 'df': df}


def select_archetypes(df: pd.DataFrame) -> pd.DataFrame:
    """Pareto front에서 aggressive / balanced / conservative 자동 선정.

    - aggressive:   max W
    - conservative: min σ_norm
    - balanced:     TOPSIS — 정규화된 (W↑, σ↓, RA↓) 이상점에 가까운 점
    """
    if len(df) == 0:
        return df.assign(archetype=[])

    F = df[['W', 'sigma_norm', 'RA']].copy()
    F['W'] = -F['W']  # NSGA-II 최소화 형태로 (W 큰 게 좋음 → 부호 반전)
    Fn = (F - F.min()) / (F.max() - F.min() + 1e-12)
    ideal = Fn.min(axis=0).values
    nadir = Fn.max(axis=0).values
    d_pos = np.linalg.norm(Fn.values - ideal, axis=1)
    d_neg = np.linalg.norm(Fn.values - nadir, axis=1)
    closeness = d_neg / (d_pos + d_neg + 1e-12)

    arche = ['' for _ in range(len(df))]
    arche[int(df['W'].idxmax())] = 'aggressive'
    arche[int(df['sigma_norm'].idxmin())] = 'conservative'
    arche[int(np.argmax(closeness))] = arche[int(np.argmax(closeness))] or 'balanced'
    out = df.copy()
    out['archetype'] = arche
    out['topsis'] = closeness
    return out


def save_pareto(df: pd.DataFrame, path: Path) -> None:
    """Pareto front DataFrame을 CSV로 저장 (재현·후속 분석용)."""
    df.to_csv(path, index=False)
