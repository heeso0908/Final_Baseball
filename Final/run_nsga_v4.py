"""NSGA-II Pareto front 재계산 v4 — 새 sim 로직 기준.

이전 v3 (random starter + 3-period closer) → v4 (fixed starter rotation + 4-period closer).
"""
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from nsga_search import run_nsga_parallel, select_archetypes, save_pareto

POP_SIZE  = 50
N_GEN     = 15
N_SEASONS = 20
N_PROC    = 6
SEED      = 42

if __name__ == '__main__':
    print(f'NSGA-II v4 (새 sim 로직 기준)')
    print(f'  pop={POP_SIZE}, gen={N_GEN}, seasons={N_SEASONS}, proc={N_PROC}')
    print(f'  시작: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    t0 = time.time()
    out = run_nsga_parallel(
        pop_size=POP_SIZE, n_gen=N_GEN, n_seasons=N_SEASONS,
        seed=SEED, n_proc=N_PROC, verbose=True,
    )
    elapsed = time.time() - t0

    df = select_archetypes(out['df'])
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = Path(__file__).parent / f'nsga_pareto_phase8_v4_{ts}.csv'
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
