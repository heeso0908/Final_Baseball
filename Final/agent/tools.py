"""TEX 2025 챗봇 도구 모음.

각 함수는 docstring(LLM이 도구 설명으로 사용)과 명확한 타입 힌트를 갖는다.
chatbot.py에서 PydanticAI Agent에 등록한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import simulation_core

_APP_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _APP_ROOT / "data_raw"
_OUTPUT_DIR = _APP_ROOT / "output"

# 사용자가 자주 쓰는 팀 호칭 → MLB 약자
_TEAM_ALIAS = {
    'mariners': 'SEA', '매리너스': 'SEA', '시애틀': 'SEA', 'sea': 'SEA',
    'astros': 'HOU', '애스트로스': 'HOU', '휴스턴': 'HOU', 'hou': 'HOU',
    'athletics': 'ATH', '애슬래틱스': 'ATH', '오클랜드': 'ATH', 'oak': 'ATH', 'ath': 'ATH',
    'angels': 'LAA', '에인절스': 'LAA', '엔젤스': 'LAA', '에인절': 'LAA', 'laa': 'LAA',
    'rangers': 'TEX', '레인저스': 'TEX', '텍사스': 'TEX', 'tex': 'TEX',
}


def _normalize_team(team: str) -> str:
    code = _TEAM_ALIAS.get(team.strip().lower(), team.strip().upper())
    return code


# ============================================================
# 1. 시나리오 잔차 점추정
# ============================================================

def estimate_residual_scenario(sigmas: dict[str, float]) -> dict:
    """σ 단위 조정으로 TEX 2025 반사실 잔차 점추정 (MC 아님).

    4모델(Ridge·Lasso·RF·XGB) 앙상블 평균으로 예상 승수를 산출하고,
    시나리오 불확실성은 4모델 std(`pred_std`)로 정량화한다.

    피처 조정 규칙: **양수 sigma = 항상 개선 방향**.
    lower-better 피처(BB9, HR9, ir_pct 등)도 양수=개선으로 통일된다.

    조정 가능 피처: sv_pct, SV_pg, onerun_wp, xi_wp, home_away_diff, WHIP,
    k_bb, K9, BB9, HR9, ir_pct, babip_against, go_ao, sb_pct, era_fip_diff.
    (ERA, OPS, rs_per_g는 pyth_W 채널과 중복되어 조정 비권장)

    Args:
        sigmas: {'K9': 0.3, 'BB9': 0.4} 같은 dict. (BB9 +0.4 = BB9 감소 = 개선)

    Returns:
        - `predicted_W_calibrated`: 사용자에게 보여줄 보정 승수 (실제 81승 기준)
        - `predicted_W_raw`: ML 점추정 (보정 전, ~86 부근)
        - `delta`: 베이스라인 잔차 대비 변화
        - `pred_std`: 4모델 예측 std (시나리오 불확실성)
        - `baseline_W_actual`: 81 (TEX 2025 실제)
        - `calibration_offset`: -5.2 부근 (ML 베이스라인 보정값)
        - `adjustments`: 피처별 σ → 원본 변화량 + new_value
        - `warnings`: 입력 검증 경고
    """
    return simulation_core.estimate_residual_scenario(sigmas)


# 하위 호환 alias
simulate_scenario = estimate_residual_scenario


# ============================================================
# 2. 사전 정의 시나리오 조회
# ============================================================

_PARETO_NAME_MAP = {
    'aggressive': '공격적 (최대 잔차 승수 개선)',
    'balanced': '균형점 (TOPSIS)',
    'conservative': '보수적 (최소 불확실성)',
    '공격적': '공격적 (최대 잔차 승수 개선)',
    '균형': '균형점 (TOPSIS)',
    '균형점': '균형점 (TOPSIS)',
    '보수적': '보수적 (최소 불확실성)',
    'topsis': '균형점 (TOPSIS)',
}

_GRID_NAMES = {'best_overall', 'best_bullpen', 'best_closegame',
               'best_pitching', 'worst_overall', 'baseline'}


def lookup_pareto(name: str) -> dict:
    """사전 계산된 Pareto/Grid 시나리오 결과 즉시 반환.

    노트북 v3 실행 시 캐시된 결과를 읽어와 시뮬 호출 없이 응답한다.

    유효한 이름:
    - Pareto 3종: 'aggressive' / 'balanced' / 'conservative' (또는 한국어)
    - Grid 6종: 'best_overall', 'best_bullpen', 'best_closegame',
      'best_pitching', 'worst_overall', 'baseline'

    Args:
        name: 시나리오 이름 (영문/한국어 alias 지원).

    Returns:
        시나리오 라벨, delta, 예상 승수, 주요 조정 피처.
    """
    key = name.strip().lower()

    # Pareto 시나리오 — v3 노트북 Cell 45 (Grid 13,824 → Pareto 필터)
    if key in _PARETO_NAME_MAP:
        path = _OUTPUT_DIR / "grid_pareto.csv"
        # 폴백: 구버전 NSGA-II 결과
        if not path.exists():
            path = _OUTPUT_DIR / "pareto_summary.csv"
        if not path.exists():
            return {'error': 'grid_pareto.csv / pareto_summary.csv 둘 다 없음. v3 노트북 Cell 45 실행 필요.'}

        df = pd.read_csv(path)
        target = _PARETO_NAME_MAP[key]
        row = df[df['유형'] == target]
        if row.empty:
            return {'error': f'행을 찾을 수 없음: {target}'}
        r = row.iloc[0]
        result = {
            'source': 'grid_pareto' if path.name == 'grid_pareto.csv' else 'nsga2',
            'label': str(r['유형']),
            'delta': round(float(r['resid_delta']), 3),
            'pred_std': round(float(r['pred_std']), 4),
            'adjustments_summary': str(r['주요 조정']),
        }
        if 'pred_W' in df.columns:
            result['predicted_W'] = round(float(r['pred_W']), 1)
        return result

    # Grid 시나리오
    if name.strip() in _GRID_NAMES:
        path = _OUTPUT_DIR / "signed_proxy_scenario_summary.csv"
        if not path.exists():
            return {'error': f'signed_proxy_scenario_summary.csv 없음 ({path}).'}
        df = pd.read_csv(path)
        row = df[df['시나리오'] == name.strip()]
        if row.empty:
            return {'error': f'시나리오 없음: {name}'}
        r = row.iloc[0]
        return {
            'source': 'grid',
            'label': str(r['시나리오']),
            'description': str(r['설명']),
            'delta': round(float(r['resid_delta']), 3),
            'mean_W': round(float(r['평균_승수']), 1),
            'median_W': round(float(r['중간값']), 1),
            'p5_W': round(float(r['P5']), 1),
            'p95_W': round(float(r['P95']), 1),
            'p_88plus_pct': round(float(r['P_88이상(%)']), 1),
            'p_90plus_pct': round(float(r['P_90이상(%)']), 1),
        }

    return {
        'error': f'알 수 없는 시나리오: {name}',
        'valid_pareto': sorted(set(_PARETO_NAME_MAP)),
        'valid_grid': sorted(_GRID_NAMES),
    }


# ============================================================
# 3. 경쟁팀 통계 비교
# ============================================================

_COMPARE_COLS = ['ERA', 'FIP', 'K/9', 'BB/9', 'HR/9', 'SV', 'BS', 'BABIP', 'GB%', 'Hard%']


def compare_team_2025(team: str) -> dict:
    """TEX 2025와 다른 팀 2025 투수 통계 비교.

    한 시즌 동시 비교만 지원한다. 양 팀 핵심 지표 + TEX 대비 차이를 반환한다.

    Args:
        team: 비교 대상 팀 (예: 'SEA', '매리너스', 'mariners', 'OAK').

    Returns:
        TEX/대상팀 핵심 통계 + 차이.
    """
    code = _normalize_team(team)
    path = _DATA_DIR / "mlb_teams_2025_pitching.csv"
    df = pd.read_csv(path)

    if code not in df['Team'].values:
        return {'error': f'팀 코드 없음: {code}', 'available_teams': sorted(df['Team'].unique())}

    tex = df[df['Team'] == 'TEX'].iloc[0]
    tgt = df[df['Team'] == code].iloc[0]

    diffs = {}
    for col in _COMPARE_COLS:
        if col not in df.columns:
            continue
        try:
            t_val = float(tex[col])
            g_val = float(tgt[col])
            diffs[col] = {
                'TEX': round(t_val, 3),
                code: round(g_val, 3),
                f'TEX-{code}': round(t_val - g_val, 3),
            }
        except (ValueError, TypeError):
            continue

    return {
        'reference_team': 'TEX',
        'compared_team': code,
        'note': '양수 = TEX가 더 큼. ERA/BB/9/HR9/BABIP는 lower-better.',
        'metrics': diffs,
    }


# ============================================================
# 4. 경쟁팀 통계 이식 시뮬
# ============================================================

def swap_team_pitching(team: str) -> dict:
    """대상 팀 2025 투수 통계를 TEX 입력에 이식해 시뮬.

    "TEX가 SEA 수준 선발진이었다면 몇 승?" 류 질문에 답한다.
    K9/BB9/HR9/BABIP 차이를 σ 단위로 변환해 estimate_residual_scenario 호출.

    Args:
        team: 이식할 팀 (예: 'SEA', '매리너스').

    Returns:
        시뮬 결과 + 이식한 σ 조정 내역.
    """
    code = _normalize_team(team)
    if code == 'TEX':
        return {'error': 'TEX는 자기 자신이라 이식 의미 없음.'}

    path = _DATA_DIR / "mlb_teams_2025_pitching.csv"
    df = pd.read_csv(path)
    if code not in df['Team'].values:
        return {'error': f'팀 코드 없음: {code}'}

    tex = df[df['Team'] == 'TEX'].iloc[0]
    tgt = df[df['Team'] == code].iloc[0]

    # 통계 컬럼 이름 매핑 (CSV → MODEL_FEATURES)
    swap_map = {
        'K/9': 'K9',
        'BB/9': 'BB9',
        'HR/9': 'HR9',
        'BABIP': 'babip_against',
    }

    info = simulation_core.feature_info()
    feat_std = info['feature_std']

    sigmas: dict[str, float] = {}
    raw_diffs: dict[str, dict] = {}
    for csv_col, feat in swap_map.items():
        if csv_col not in df.columns:
            continue
        raw_diff = float(tgt[csv_col]) - float(tex[csv_col])
        sigma_raw = raw_diff / feat_std[feat] if feat_std[feat] > 0 else 0.0
        # 부호 변환: 모듈 simulate_scenario는 양수=개선 규칙.
        # raw_diff > 0 = 대상팀이 더 큼.
        # lower-better 피처(BB9/HR9/babip): 대상팀이 크면 = TEX가 (상대적으로) 더 좋음 → 이식 시 악화 → σ 음수
        # higher-better(K9): 대상팀이 크면 = TEX 이식 시 개선 → σ 양수
        if feat in {'BB9', 'HR9', 'babip_against'}:
            sigma = -sigma_raw  # 대상팀 값으로 바꾸면 (대상팀-TEX) 크기만큼 악화
        else:
            sigma = sigma_raw
        sigmas[feat] = round(sigma, 3)
        raw_diffs[feat] = {
            'TEX': round(float(tex[csv_col]), 3),
            code: round(float(tgt[csv_col]), 3),
            'sigma_applied': round(sigma, 3),
        }

    sim_result = simulation_core.estimate_residual_scenario(sigmas)
    sim_result['source_team'] = code
    sim_result['raw_swap'] = raw_diffs
    sim_result['caveat'] = (
        '통계만 swap한 ceiling 추정. 라인업/구장/부상 효과는 미반영.'
    )
    return sim_result


# ============================================================
# 5. TEX 게임로그 조회
# ============================================================

def query_gamelog(
    month: int | None = None,
    opponent: str | None = None,
    home_only: bool = False,
    away_only: bool = False,
    one_run_only: bool = False,
    extra_innings_only: bool = False,
) -> dict:
    """TEX 2025 게임로그 필터 조회.

    조건에 맞는 경기들의 승률, 평균 득실 등을 집계한다.

    Args:
        month: 1~12 월 필터.
        opponent: 상대팀 코드 (예: 'SEA').
        home_only / away_only: 홈/원정 필터.
        one_run_only: 1점차 경기만.
        extra_innings_only: 연장전만.

    Returns:
        경기 수, 승/패, 평균 득점, 평균 실점, 승률.
    """
    path = _DATA_DIR / "texas_2025_game_log.csv"
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
    df = df.dropna(subset=['Date'])

    mask = pd.Series(True, index=df.index)
    if month is not None:
        mask &= df['Date'].dt.month == month
    if opponent is not None:
        mask &= df['Opp'] == _normalize_team(opponent)
    if home_only:
        mask &= df['Home_Away'] == 'Home'
    if away_only:
        mask &= df['Home_Away'] == 'Away'
    if one_run_only:
        mask &= (df['R'] - df['RA']).abs() == 1
    if extra_innings_only:
        mask &= df['Inn'] > 9

    sub = df[mask]
    if sub.empty:
        return {'matched_games': 0, 'note': '조건에 맞는 경기 없음.'}

    wins = (sub['W/L'].astype(str).str.startswith('W')).sum()
    losses = len(sub) - wins
    return {
        'matched_games': int(len(sub)),
        'wins': int(wins),
        'losses': int(losses),
        'win_pct': round(float(wins) / len(sub), 3),
        'avg_R': round(float(sub['R'].mean()), 2),
        'avg_RA': round(float(sub['RA'].mean()), 2),
        'run_diff_total': int(sub['R'].sum() - sub['RA'].sum()),
    }


# ============================================================
# 6. 10개년 historical 조회
# ============================================================

def query_team_history(
    year_from: int | None = None,
    year_to: int | None = None,
    residual_min: float | None = None,
    residual_max: float | None = None,
    team: str | None = None,
    top_n: int = 10,
) -> dict:
    """10개년 MLB 팀 시즌에서 조건 매칭 팀 조회.

    "역대 잔차 -9승 수준 팀이 있었어?", "TEX 역대 잔차 추이는?" 같은
    historical 비교 질문에 답한다.

    Args:
        year_from / year_to: 연도 범위 (포함).
        residual_min / residual_max: 잔차(W - pyth_W) 범위.
        team: 특정 팀만 (예: 'TEX').
        top_n: 반환할 행 수 상한.

    Returns:
        매칭된 팀-시즌 리스트 + 요약 통계.
    """
    path = _DATA_DIR / "mlb_team_seasons.csv"
    df = pd.read_csv(path)
    df['pyth_wp'] = df['RS']**1.83 / (df['RS']**1.83 + df['RA']**1.83)
    df['pyth_W'] = (df['pyth_wp'] * df['G']).round(1)
    df['residual'] = df['W'] - df['pyth_W']

    mask = pd.Series(True, index=df.index)
    if year_from is not None:
        mask &= df['year'] >= year_from
    if year_to is not None:
        mask &= df['year'] <= year_to
    if residual_min is not None:
        mask &= df['residual'] >= residual_min
    if residual_max is not None:
        mask &= df['residual'] <= residual_max
    if team is not None:
        # team 파일은 풀네임. team_id 또는 약자 모두 받자.
        code = _normalize_team(team)
        # mlb_team_seasons.csv는 'team' 컬럼이 풀네임 (예: 'Texas Rangers')
        mask &= df['team'].str.contains(code, case=False, na=False) | (df['team'].str[:3].str.upper() == code)

    sub = df[mask].copy()
    if sub.empty:
        return {'matched': 0, 'note': '조건에 맞는 팀-시즌 없음.'}

    sub = sub.sort_values('residual')
    rows = sub[['year', 'team', 'W', 'pyth_W', 'residual']].head(top_n)
    return {
        'matched': int(len(sub)),
        'returned': int(len(rows)),
        'mean_residual': round(float(sub['residual'].mean()), 2),
        'rows': rows.to_dict(orient='records'),
    }
