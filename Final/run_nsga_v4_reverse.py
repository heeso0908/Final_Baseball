"""역방향 NSGA-II — 잔차 -9 mechanical signature 탐색.

목적함수 (2-objective, minimize):
    1. |W - 81|     → 실제 W=81에 얼마나 가까운가
    2. σ_norm       → 가능한 작은 stat 변화로 잔차 재현

Pareto front 중 σ_norm 가장 작으면서 |W-81|≈0인 점이 잔차의 *mechanical signature* 후보.
그 σ 패턴 (어느 stat을 얼마나 약화/강화시켰을 때 W=81 재현) → TEX 2025 실제 stat 약점과 비교.
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd

from nsga_search import (
    SIGMA_DIMS, N_VAR, LOWER, UPPER, DIM_IDS,
    evaluate_sigma, sigma_norm, _build_algorithm,
)
from pymoo.core.problem import ElementwiseProblem
from pymoo.parallelization import StarmapParallelization
from pymoo.optimize import minimize
from pymoo.termination import get_termination
from integrated_sim import _ensure_loaded


TARGET_W  = 81.0
POP_SIZE  = 40
N_GEN     = 12
N_SEASONS = 20
N_PROC    = 6
SEED      = 42


class TexSigmaReverseProblem(ElementwiseProblem):
    """역방향 — 2-objective: [|W - target_W|, σ_norm]."""

    def __init__(self, n_seasons=20, target_W=81.0, **kw):
        super().__init__(n_var=N_VAR, n_obj=2, xl=LOWER, xu=UPPER, **kw)
        self.n_seasons = n_seasons
        self.target_W = target_W

    def _evaluate(self, x, out, *args, **kwargs):
        ev = evaluate_sigma(x, n_seasons=self.n_seasons)
        out['F'] = [abs(ev['W'] - self.target_W), sigma_norm(x)]


def _result_to_df(res):
    X = res.X if res.X is not None else np.empty((0, N_VAR))
    F = res.F if res.F is not None else np.empty((0, 2))
    cols = {'abs_W_diff': F[:, 0], 'sigma_norm': F[:, 1]}
    # W 재계산 (F[0]은 |W-81|이라 부호 모름 — 평균 W 다시 평가)
    Ws, RSs, RAs = [], [], []
    for i in range(len(X)):
        ev = evaluate_sigma(X[i], n_seasons=N_SEASONS)
        Ws.append(ev['W'])
        RSs.append(ev['RS'])
        RAs.append(ev['RA'])
    cols['W'] = Ws
    cols['RS'] = RSs
    cols['RA'] = RAs
    for i, did in enumerate(DIM_IDS):
        cols[did] = X[:, i]
    return X, F, pd.DataFrame(cols)


if __name__ == '__main__':
    import multiprocessing as mp

    print(f'NSGA-II 역방향 (잔차 -9 mechanical signature)')
    print(f'  target W = {TARGET_W}, pop={POP_SIZE}, gen={N_GEN}, seasons={N_SEASONS}, proc={N_PROC}')
    print(f'  시작: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    _ensure_loaded()
    t0 = time.time()

    ctx = mp.get_context('spawn')
    with ctx.Pool(N_PROC) as pool:
        runner = StarmapParallelization(pool.starmap)
        problem = TexSigmaReverseProblem(
            n_seasons=N_SEASONS, target_W=TARGET_W,
            elementwise_runner=runner,
        )
        algorithm = _build_algorithm(POP_SIZE)
        termination = get_termination('n_gen', N_GEN)
        res = minimize(problem, algorithm, termination, seed=SEED,
                       verbose=True, save_history=False)

    elapsed = time.time() - t0
    print(f'\n최적화 완료: {elapsed/60:.1f}분')
    print(f'결과 점수: {len(res.X) if res.X is not None else 0}')

    print('\nPareto 점들 W 재평가 중...')
    X, F, df = _result_to_df(res)

    # σ_norm 작은 순으로 정렬 (signature 후보 우선)
    df = df.sort_values('sigma_norm').reset_index(drop=True)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(__file__).parent / f'nsga_reverse_signature_{ts}.csv'
    df.to_csv(out_path, index=False)

    print(f'\n저장: {out_path.name}')
    print(f'  σ_norm 범위: [{df["sigma_norm"].min():.3f}, {df["sigma_norm"].max():.3f}]')
    print(f'  W 범위:      [{df["W"].min():.1f}, {df["W"].max():.1f}]')
    print(f'  |W-81| 범위: [{df["abs_W_diff"].min():.2f}, {df["abs_W_diff"].max():.2f}]')
    print()

    # signature 후보: |W-81| 작으면서 σ_norm 작은 점
    sig_zone = df[df['abs_W_diff'] < 1.5].sort_values('sigma_norm').head(5)
    print('=== 잔차 mechanical signature 후보 (|W-81| < 1.5, σ_norm 작은 순 5점) ===')
    cols_show = ['W', 'sigma_norm', 'RA'] + DIM_IDS
    print(sig_zone[cols_show].to_string(index=False, float_format='%.3f'))
