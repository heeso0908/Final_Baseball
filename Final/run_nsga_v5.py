"""NSGA-II Pareto front v5 — σ 범위 현실 분포 안으로 좁힌 후 재실행 (2026-05-12).

v4 (이전 σ 범위) → v5 (좁은 범위, 외삽 영역 제거).
- 타자: ±15% (이전 ±30%)
- 투수 closer: ±15% (이전 -50% ~ +30%)
- 공통: ±10%

다른 설정은 v4와 동일 (pop=50, gen=15, seasons=20, proc=6).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nsga_search import run_nsga_parallel, select_archetypes, save_pareto, SIGMA_DIMS

POP_SIZE  = 50
N_GEN     = 15
N_SEASONS = 20
N_PROC    = 6
SEED      = 42

if __name__ == '__main__':
    print(f'NSGA-II v5 (좁은 σ 범위, 외삽 제거)')
    print(f'  pop={POP_SIZE}, gen={N_GEN}, seasons={N_SEASONS}, proc={N_PROC}')
    print(f'  σ 범위:')
    for d in SIGMA_DIMS:
        print(f'    {d["id"]:10s}  [{d["lo"]:.2f}, {d["hi"]:.2f}]  ({d["group"]}.{d["sub"]}.{d["key"]})')
    print(f'  시작: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    t0 = time.time()
    out = run_nsga_parallel(
        pop_size=POP_SIZE, n_gen=N_GEN, n_seasons=N_SEASONS,
        seed=SEED, n_proc=N_PROC, verbose=True,
    )
    elapsed = time.time() - t0

    df = select_archetypes(out['df'])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(__file__).parent / f'nsga_pareto_phase8_v5_{ts}.csv'
    save_pareto(df, out_path)

    print()
    print(f'완료: {elapsed/60:.1f}분')
    print(f'Pareto front: {len(df)}점')
    print(f'  W 범위: [{df["W"].min():.1f}, {df["W"].max():.1f}]')
    print(f'  σ_norm 범위: [{df["sigma_norm"].min():.3f}, {df["sigma_norm"].max():.3f}]')
    print(f'  저장: {out_path.name}')
    print()
    print('archetype 후보:')
    print(df[df['archetype'] != ''][['archetype', 'W', 'sigma_norm', 'RA', 'topsis']].to_string(index=False))
