"""TEX 통합 시뮬: v5 simulator(타자 markov) + markov_pitching(투수 markov + bullpen).

Phase 8 NSGA-II σ 시나리오 진입점. v5 노트북 cell #61-63의 통합 로직을 모듈로 추출.

Usage:
    from integrated_sim import run_integrated_simulation

    df, monthly_df = run_integrated_simulation(
        n_seasons=100,
        hitter_adjustments={'team': {'hr_mult': 1.10, 'k_mult': 0.95}},
        pitcher_adjustments={'closer': {'K_pct': 1.10}, 'common': {'team_BB_pct_mult': 0.95}},
    )
    print(df[['W', 'RS', 'RA']].mean())

    # 선수별 기록 추적
    df, monthly_df, batter_df, pitcher_df = run_integrated_simulation(
        n_seasons=50, track_player_stats=True
    )
"""
from __future__ import annotations

import bisect
from pathlib import Path
import numpy as np
import pandas as pd

from markov_pitching import simulate_tex_RA, simulate_inning_RA, get_pitcher_name
from simulator import (
    _train_model_bundle, _build_batting_pool, _build_advancement_tables,
    _advance_runners, _simulate_inning as _batting_inning,
    LineupPool, _simulate_texas_runs,
)

_APP_ROOT = Path(__file__).resolve().parent
_DEFAULT_RAW_DIR = _APP_ROOT / "data_raw"

# module-level lazy 캐시 — bundle/batting_pool/advancement 만 보관 (LineupPool은 시나리오마다 새로)
_state: dict = {}

# 타자 이벤트 목록 (simulator._extract_probs 기준)
_BATTER_EVENTS = ('single', 'double', 'triple', 'hr', 'bb', 'hbp', 'k', 'groundout', 'flyout')

# TEX 2025 실제 연장전 승률: 9W-8L (17경기)
TEX_EXTRA_INN_WIN_RATE: float = 9 / 17

# 월 표시 순서
_MONTH_ORDER = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']


def _load_game_months(raw_dir: Path) -> list[str]:
    """game_idx(0-161) → 월 약어(Apr 등) 리스트. game log에서 추출."""
    path = raw_dir / 'texas_2025_game_log.csv'
    if not path.exists():
        # fallback: 월별 27경기 균등 배분
        months = ['Mar']*5 + ['Apr']*27 + ['May']*27 + ['Jun']*27 + \
                 ['Jul']*27 + ['Aug']*27 + ['Sep']*22
        return months[:162]
    gl = pd.read_csv(path)
    gl['R'] = pd.to_numeric(gl.get('R', pd.Series(dtype=float)), errors='coerce')
    gl = gl.dropna(subset=['R']).reset_index(drop=True)
    months = gl['Date'].str.extract(r',\s*(\w+)\s+\d+')[0].tolist()
    return months[:162]


def _ensure_loaded(raw_dir: Path = _DEFAULT_RAW_DIR) -> None:
    if 'batting_pool' in _state and _state.get('raw_dir') == raw_dir:
        return
    bundle = _train_model_bundle(raw_dir)
    batting_pool, sprint_map, hbp_df = _build_batting_pool(raw_dir)
    advancement = _build_advancement_tables(raw_dir)
    _state.update({
        'raw_dir':      raw_dir,
        'bundle':       bundle,
        'batting_pool': batting_pool,
        'sprint_map':   sprint_map,
        'hbp_df':       hbp_df,
        'advancement':  advancement,
        'game_months':  _load_game_months(raw_dir),
    })


def _build_lineup_pool(hitter_adjustments: dict | None) -> LineupPool:
    """hitter_adjustments → LineupPool 인자로 풀어서 생성."""
    name_boosts = hitter_adjustments.get('per_player') if hitter_adjustments else None
    team_boosts = hitter_adjustments.get('team')        if hitter_adjustments else None
    return LineupPool(
        _state['batting_pool'], _state['sprint_map'], _state['hbp_df'],
        name_boosts=name_boosts, team_boosts=team_boosts,
    )


# ──────────────────────────────────────────────────
# 타자 추적 버전
# ──────────────────────────────────────────────────

def _make_batter_stat() -> dict:
    return {e: 0 for e in _BATTER_EVENTS}


def _simulate_inning_tracked(
    lineup_probs: list,
    profiles: list,
    selected: list,
    start_index: int,
    advancement=None,
) -> tuple[int, int, dict]:
    """_simulate_inning()의 선수별 추적 버전.

    Returns:
        (runs, next_lineup_index, {player_name: {event: count}})
    """
    occupied: list = []
    outs = 0
    runs = 0
    hitter_index = start_index
    lineup_size = len(lineup_probs)
    batter_events: dict = {}

    while outs < 3:
        slot = hitter_index % lineup_size
        name = selected[slot]
        _evts, _cum = lineup_probs[slot]
        event = _evts[bisect.bisect(_cum, np.random.random())]
        occupied, outs, scored = _advance_runners(
            occupied, outs, event, profiles[slot], advancement=advancement
        )
        runs += scored
        hitter_index += 1

        s = batter_events.setdefault(name, _make_batter_stat())
        s[event] = s.get(event, 0) + 1

    return runs, hitter_index % lineup_size, batter_events


def _simulate_texas_runs_tracked(
    lineup_probs: list,
    profiles: list,
    selected: list,
    advancement=None,
) -> tuple[int, dict]:
    """_simulate_texas_runs()의 선수별 추적 버전.

    Returns:
        (total_runs, {player_name: {event: count}})
    """
    total_runs = 0
    lineup_index = 0
    all_batter_stats: dict = {}

    for _ in range(9):
        inning_runs, lineup_index, batter_events = _simulate_inning_tracked(
            lineup_probs, profiles, selected, lineup_index, advancement=advancement
        )
        total_runs += inning_runs
        for name, evts in batter_events.items():
            s = all_batter_stats.setdefault(name, _make_batter_stat())
            for k, v in evts.items():
                s[k] = s.get(k, 0) + v

    return total_runs, all_batter_stats


# ──────────────────────────────────────────────────
# 결과 DataFrame 빌더
# ──────────────────────────────────────────────────

def _build_batter_df(all_batter: dict, n_seasons: int) -> pd.DataFrame:
    """누적 타자 이벤트 → 시즌 평균 성적 DataFrame."""
    rows = []
    for name, stats in all_batter.items():
        s1  = stats.get('single', 0)
        s2  = stats.get('double', 0)
        s3  = stats.get('triple', 0)
        hr  = stats.get('hr', 0)
        bb  = stats.get('bb', 0)
        hbp = stats.get('hbp', 0)
        k   = stats.get('k', 0)
        go  = stats.get('groundout', 0)
        fo  = stats.get('flyout', 0)
        pa  = s1 + s2 + s3 + hr + bb + hbp + k + go + fo
        h   = s1 + s2 + s3 + hr
        ab  = max(pa - bb - hbp, 1)
        avg = h / ab
        obp = (h + bb + hbp) / pa if pa > 0 else 0.0
        slg = (s1 + 2 * s2 + 3 * s3 + 4 * hr) / ab
        rows.append({
            '선수':    name,
            'PA/시즌': round(pa / n_seasons, 1),
            'H/시즌':  round(h / n_seasons, 1),
            '2B/시즌': round(s2 / n_seasons, 1),
            '3B/시즌': round(s3 / n_seasons, 1),
            'HR/시즌': round(hr / n_seasons, 1),
            'BB/시즌': round(bb / n_seasons, 1),
            'K/시즌':  round(k / n_seasons, 1),
            'AVG':     round(avg, 3),
            'OBP':     round(obp, 3),
            'SLG':     round(slg, 3),
            'OPS':     round(obp + slg, 3),
        })
    return (
        pd.DataFrame(rows)
        .sort_values('OPS', ascending=False)
        .reset_index(drop=True)
    )


def _build_pitcher_df(all_pitcher: dict, n_seasons: int) -> pd.DataFrame:
    """누적 투수 이벤트 → 시즌 평균 성적 DataFrame."""
    rows = []
    for pid, stats in all_pitcher.items():
        bf   = stats.get('BF', 0)
        k    = stats.get('K', 0)
        bb   = stats.get('BB', 0)
        hr   = stats.get('HR', 0)
        h    = stats.get('H', 0)
        outs = stats.get('outs', 0)
        er   = stats.get('ER', 0)
        ip   = (outs + k) / 3.0  # K도 아웃으로 계산
        era  = (er / max(ip, 0.1)) * 9
        whip = (h + hr + bb) / max(ip, 0.1)  # WHIP = (1B+2B+3B+HR+BB) / IP
        rows.append({
            '투수':    get_pitcher_name(pid),
            'IP/시즌': round(ip / n_seasons, 1),
            'BF/시즌': round(bf / n_seasons, 1),
            'K/시즌':  round(k / n_seasons, 1),
            'BB/시즌': round(bb / n_seasons, 1),
            'HR/시즌': round(hr / n_seasons, 1),
            'H/시즌':  round(h / n_seasons, 1),
            'ER/시즌': round(er / n_seasons, 1),
            'ERA':     round(era, 2),
            'WHIP':    round(whip, 2),
            'K%':      round(k / max(bf, 1), 3),
            'BB%':     round(bb / max(bf, 1), 3),
        })
    return (
        pd.DataFrame(rows)
        .sort_values('ERA')
        .reset_index(drop=True)
    )


# ──────────────────────────────────────────────────
# 핵심 시뮬 함수
# ──────────────────────────────────────────────────

def simulate_game_integrated(
    rng: np.random.Generator,
    lineup_pool: LineupPool,
    game_idx: int,
    pitcher_adjustments: dict | None = None,
    track_stats: bool = False,
) -> tuple:
    """1경기 시뮬 → (RS, RA, W, batter_stats, pitcher_stats).

    이닝별 연동: 매 이닝 공격(타자 Markov) 후 해당 시점 점수차로 수비(투수 Markov) 실행.
    동점 시 TEX 2025 실제 연장전 승률(9/17 ≈ 0.529)로 확률 샘플링.
    track_stats=False이면 batter_stats, pitcher_stats은 None.
    """
    lineup_probs, profiles, selected = lineup_pool.sample()
    advancement = _state['advancement']

    tex_rs = 0
    tex_ra = 0
    lineup_index = 0
    used_closer_flag = [False]
    all_batter_stats: dict | None = {} if track_stats else None
    pitcher_game_stats: dict | None = {} if track_stats else None

    for inning in range(1, 10):
        # TEX 공격 — 타자 Markov (이닝 단위)
        if track_stats:
            inning_rs, lineup_index, batter_events = _simulate_inning_tracked(
                lineup_probs, profiles, selected, lineup_index, advancement=advancement
            )
            for name, evts in batter_events.items():
                s = all_batter_stats.setdefault(name, _make_batter_stat())
                for k, v in evts.items():
                    s[k] = s.get(k, 0) + v
        else:
            inning_rs, lineup_index = _batting_inning(
                lineup_probs, profiles, lineup_index, advancement=advancement
            )
        tex_rs += inning_rs

        # TEX 수비 — 투수 Markov (실시간 점수차 반영)
        score_diff = tex_rs - tex_ra
        inning_ra = simulate_inning_RA(
            inning, score_diff, used_closer_flag, rng,
            game_idx=game_idx, pitcher_adjustments=pitcher_adjustments,
            pitcher_game_stats=pitcher_game_stats,
        )
        tex_ra += inning_ra

    if tex_rs > tex_ra:
        w = 1.0
    elif tex_rs == tex_ra:
        # 실제 연장전 승률로 확률 샘플링 (TEX 2025: 9W-8L)
        w = 1.0 if rng.random() < TEX_EXTRA_INN_WIN_RATE else 0.0
    else:
        w = 0.0

    return tex_rs, tex_ra, w, all_batter_stats, pitcher_game_stats


def run_integrated_season(
    seed: int,
    lineup_pool: LineupPool,
    pitcher_adjustments: dict | None = None,
    n_games: int = 162,
    track_stats: bool = False,
) -> dict:
    """단일 시즌(162경기) 시뮬 → {W, RS, RA, monthly_w, batter_stats, pitcher_stats}."""
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    rs_total, ra_total, w_total = 0, 0, 0.0
    season_batter: dict = {}
    season_pitcher: dict = {}
    game_months = _state.get('game_months', [])
    monthly_w: dict[str, float] = {}

    for game_idx in range(n_games):
        rs, ra, w, bat, pit = simulate_game_integrated(
            rng, lineup_pool, game_idx,
            pitcher_adjustments=pitcher_adjustments,
            track_stats=track_stats,
        )
        rs_total += rs
        ra_total += ra
        w_total  += w

        # 월별 승수 추적
        month = game_months[game_idx] if game_idx < len(game_months) else 'Sep'
        monthly_w[month] = monthly_w.get(month, 0.0) + w

        if track_stats and bat:
            for name, evts in bat.items():
                s = season_batter.setdefault(name, _make_batter_stat())
                for k, v in evts.items():
                    s[k] = s.get(k, 0) + v

        if track_stats and pit:
            for pid, stats in pit.items():
                s = season_pitcher.setdefault(
                    pid, {'BF': 0, 'K': 0, 'BB': 0, 'HR': 0, 'H': 0, 'outs': 0, 'ER': 0}
                )
                for k, v in stats.items():
                    s[k] += v

    return {
        'W': w_total, 'RS': rs_total, 'RA': ra_total,
        'monthly_w':     monthly_w,
        'batter_stats':  season_batter  if track_stats else None,
        'pitcher_stats': season_pitcher if track_stats else None,
    }


def run_integrated_simulation(
    n_seasons: int = 100,
    hitter_adjustments: dict | None = None,
    pitcher_adjustments: dict | None = None,
    n_games: int = 162,
    raw_dir: Path = _DEFAULT_RAW_DIR,
    seeds=None,
    progress: bool = False,
    track_player_stats: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """N개 시즌 시뮬 → DataFrame[W, RS, RA].

    track_player_stats=True이면 (season_df, batter_df, pitcher_df) 튜플 반환.

    Args:
        n_seasons: 시즌 수
        hitter_adjustments: {'team': {...}, 'per_player': {...}}
        pitcher_adjustments: markov_pitching σ 시나리오 dict
        n_games: 시즌당 경기 수 (default 162)
        raw_dir: data_raw 디렉토리 경로
        seeds: 명시적 seed iterable, None이면 range(n_seasons)
        progress: tqdm 진행률 표시
        track_player_stats: True이면 선수별 기록 집계 후 DataFrame으로 반환
    """
    _ensure_loaded(raw_dir)
    lineup_pool = _build_lineup_pool(hitter_adjustments)

    seed_iter = list(seeds) if seeds is not None else list(range(n_seasons))
    if progress:
        from tqdm import tqdm
        seed_iter = tqdm(seed_iter, desc='integrated sim')

    rows = [
        run_integrated_season(s, lineup_pool, pitcher_adjustments, n_games,
                              track_stats=track_player_stats)
        for s in seed_iter
    ]

    season_df = pd.DataFrame([
        {k: v for k, v in r.items()
         if k not in ('batter_stats', 'pitcher_stats', 'monthly_w')}
        for r in rows
    ])

    # 월별 승수 집계 (항상 수행)
    all_monthly: dict[str, list] = {}
    for row in rows:
        for month, w in row.get('monthly_w', {}).items():
            all_monthly.setdefault(month, []).append(w)

    month_rows = []
    for month in _MONTH_ORDER:
        ws = all_monthly.get(month, [])
        if ws:
            month_rows.append({
                'month':     month,
                'mean_wins': float(np.mean(ws)),
                'p25_wins':  float(np.percentile(ws, 25)),
                'p75_wins':  float(np.percentile(ws, 75)),
            })
    monthly_df = pd.DataFrame(month_rows)

    if not track_player_stats:
        return season_df, monthly_df

    # 전체 시즌에 걸쳐 선수별 이벤트 집계
    all_batter: dict = {}
    all_pitcher: dict = {}
    for row in rows:
        if row['batter_stats']:
            for name, stats in row['batter_stats'].items():
                s = all_batter.setdefault(name, _make_batter_stat())
                for k, v in stats.items():
                    s[k] = s.get(k, 0) + v
        if row['pitcher_stats']:
            for pid, stats in row['pitcher_stats'].items():
                s = all_pitcher.setdefault(
                    pid, {'BF': 0, 'K': 0, 'BB': 0, 'HR': 0, 'H': 0, 'outs': 0, 'ER': 0}
                )
                for k, v in stats.items():
                    s[k] += v

    batter_df  = _build_batter_df(all_batter, n_seasons)
    pitcher_df = _build_pitcher_df(all_pitcher, n_seasons)

    return season_df, monthly_df, batter_df, pitcher_df


def get_loaded_state() -> dict:
    """디버깅·검증용."""
    _ensure_loaded()
    return _state
