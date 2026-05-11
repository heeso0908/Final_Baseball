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

_SCENARIO_ROWS = [
    # Phase 8: 12차원 σ NSGA-II 시뮬 직접 평가 결과 (현실 권장 zone: σ_norm ≤ 0.10)
    {"key": "phase8_max", "source": "Phase 8", "label": "Phase 8: 잔차 초과 달성 (σ=7.7%)", "delta": 15.7, "predicted_W": 96.7, "sigma_norm": 0.077, "rank": 1, "adjustments_summary": "h_single:+13.6%, h_k:-8.0%, p_st_HR:-2.6%, c_K:+7.5%", "decision_note": "시뮬 직접 평가 (12차원 σ): 피타고리안 기대 승수 초과 달성. 타자 단타+K 감소 주도"},
    {"key": "phase8_recovery", "source": "Phase 8", "label": "Phase 8: 잔차 만회 기준 (σ=8.1%)", "delta": 9.5, "predicted_W": 90.5, "sigma_norm": 0.081, "rank": 2, "adjustments_summary": "h_single:+18.1%, p_st_HR:-4.0%, c_K:+7.9%", "decision_note": "시뮬 직접 평가 (12차원 σ): 잔차 -9 만회 수준. 타자 단타 증가·선발 HR 감소 조합"},
    {"key": "phase8_safe", "source": "Phase 8", "label": "Phase 8: 소폭 개선 (σ=6.0%)", "delta": 3.65, "predicted_W": 84.65, "sigma_norm": 0.060, "rank": 3, "adjustments_summary": "h_single:+13.6%, h_hr:+11.5%, p_st_HR:-2.4%", "decision_note": "시뮬 직접 평가 (12차원 σ): 최소 정책 변경으로 현실적 소폭 개선"},
    # v5 ML Pareto (6차원 잔차 보정 모델)
    {"key": "pareto_aggressive", "source": "Pareto", "label": "공격적 (std=0.652)", "delta": 4.889, "predicted_W": 85.9, "pred_std": 0.6524, "rank": 4, "adjustments_summary": "sv_pct:+0.107, ir_pct:+0.067, onerun_wp:+0.131, xi_wp:+0.220, HR9:-0.265, BB9:-0.582", "decision_note": "상한선 후보: 개선 가능성은 가장 크지만 불확실성 확인 필요"},
    {"key": "pareto_balanced", "source": "Pareto", "label": "균형점 (std=0.205)", "delta": 3.720, "predicted_W": 84.7, "pred_std": 0.2047, "rank": 5, "adjustments_summary": "sv_pct:+0.032, ir_pct:+0.067, onerun_wp:+0.115, xi_wp:+0.220, HR9:-0.271, BB9:-0.562", "decision_note": "균형 후보: 개선 폭과 안정성의 기본 검토안"},
    {"key": "pareto_conservative", "source": "Pareto", "label": "보수적 (std=0.002)", "delta": 2.919, "predicted_W": 83.9, "pred_std": 0.0019, "rank": 6, "adjustments_summary": "sv_pct:+0.045, ir_pct:+0.007, onerun_wp:+0.109, xi_wp:+0.176, HR9:-0.102, BB9:+0.099", "decision_note": "안정 후보: 모델 간 의견 차이가 작을 때 참고"},
    {"key": "manual_baseline", "source": "수동", "label": "Baseline 2025", "delta": 0.000, "predicted_W": 81.0, "rank": 7, "decision_note": "baseline과 차이가 작아 우선순위 낮음"},
]

_SCENARIO_BY_KEY = {row["key"]: row for row in _SCENARIO_ROWS}

_OPTIMIZATION_ROWS = [
    {
        "key": "nsga2_max",
        "source": "Pareto(최대)",
        "label": "NSGA-II Pareto 최대",
        "delta": 4.889,
        "predicted_W": 85.9,
        "pred_std": 0.652,
        "search_count": "pop x gen",
        "runtime": "~2분",
        "space": "sigma in [-1.5, 1.5], 다목적",
        "note": "residual delta 최대화와 pred_std 최소화를 동시에 고려.",
    },
]

_OPTIMIZATION_BY_KEY = {row["key"]: row for row in _OPTIMIZATION_ROWS}

_SCENARIO_ALIASES = {
    "aggressive": "pareto_aggressive",
    "attack": "pareto_aggressive",
    "pareto_aggressive": "pareto_aggressive",
    "공격적": "pareto_aggressive",
    "pareto 공격적": "pareto_aggressive",
    "balanced": "pareto_balanced",
    "balance": "pareto_balanced",
    "topsis": "pareto_balanced",
    "pareto_balanced": "pareto_balanced",
    "균형": "pareto_balanced",
    "균형점": "pareto_balanced",
    "pareto 균형점": "pareto_balanced",
    "conservative": "pareto_conservative",
    "pareto_conservative": "pareto_conservative",
    "보수적": "pareto_conservative",
    "pareto 보수적": "pareto_conservative",
    "bullpen_sv70": "manual_bullpen_upgrade",
    "manual_bullpen_sv70": "manual_bullpen_upgrade",
    "manual_bullpen_upgrade": "manual_bullpen_upgrade",
    "bullpen upgrade": "manual_bullpen_upgrade",
    "불펜 강화": "manual_bullpen_upgrade",
    "불펜 강화 (sv% 70%)": "manual_bullpen_upgrade",
    "manual_baseline": "manual_baseline",
    "기준선": "manual_baseline",
    "기준선 (2025 실제)": "manual_baseline",
    "hitter": "hitter_boost",
    "hitter_boost": "hitter_boost",
    "manual_hitter_boost": "hitter_boost",
    "hitter boost": "hitter_boost",
    "타자": "hitter_boost",
    "타자 강화": "hitter_boost",
    "타자 스탯 상승": "hitter_boost",
    "langford": "hitter_boost",
    "manual_langford": "hitter_boost",
    "langford 도약": "hitter_boost",
    "랭포드 도약": "hitter_boost",
    "nsga2": "nsga2_max",
    "nsga-ii": "nsga2_max",
    "pareto max": "nsga2_max",
    "pareto 최대": "nsga2_max",
    "nsga2_max": "nsga2_max",
    # Phase 8 시뮬 12차원 σ
    "phase8_max": "phase8_max",
    "phase8 max": "phase8_max",
    "phase8 잔차 초과": "phase8_max",
    "phase8_recovery": "phase8_recovery",
    "phase8 recovery": "phase8_recovery",
    "phase8 잔차 만회": "phase8_recovery",
    "phase8_safe": "phase8_safe",
    "phase8 safe": "phase8_safe",
    "phase8 소폭": "phase8_safe",
}


def lookup_pareto(name: str) -> dict:
    """사전 계산된 v5 통합 비교 시나리오를 즉시 반환.

    기존 챗봇 호환성을 위해 함수명은 `lookup_pareto`로 유지하지만,
    Pareto 3종과 Phase 8 시나리오, 수동 시나리오도 함께 조회한다.
    """
    raw = name.strip()
    key = _SCENARIO_ALIASES.get(raw.lower(), _SCENARIO_ALIASES.get(raw, raw))
    row = _SCENARIO_BY_KEY.get(key)
    if row is None and key in _OPTIMIZATION_BY_KEY:
        row = _OPTIMIZATION_BY_KEY[key]
    if row is None:
        return {
            "error": f"알 수 없는 시나리오: {name}",
            "valid": sorted(_SCENARIO_ALIASES),
        }

    result = dict(row)
    result["predicted_W"] = round(float(result["predicted_W"]), 1)
    result["mean_W"] = result["predicted_W"]
    result["delta"] = round(float(result["delta"]), 3)
    if "pred_std" in result:
        result["pred_std"] = round(float(result["pred_std"]), 3)
    return result


def list_precomputed_scenarios() -> dict:
    """v5 통합 비교 시나리오 전체 목록을 rank 순서로 반환."""
    def decision_note(row: dict) -> str:
        scen = str(row.get("label", ""))
        src = str(row.get("source", ""))
        delta = float(row.get("delta", 0.0))
        if src == "Phase 8":
            return row.get("decision_note", "시뮬 12차원 σ 직접 평가 후보")
        if src == "Pareto" and "공격적" in scen:
            return "상한선 후보: 개선 가능성은 가장 크지만 불확실성 확인 필요"
        if src == "Pareto" and "균형점" in scen:
            return "균형 후보: 개선 폭과 안정성의 기본 비교점"
        if src == "Pareto" and "보수적" in scen:
            return "안정 후보: 모델 간 의견 차이가 작을 때 참고"
        if src == "수동" and ("희망" in scen or "Hopeful" in scen):
            return "야구적으로 설명 가능한 현실적 개선 후보"
        if src == "수동" and ("비관" in scen or "부상" in scen):
            return "부상·슬럼프 악화 가능성 점검"
        if abs(delta) < 0.05:
            return "baseline과 차이가 작아 우선순위 낮음"
        return "보조 비교 후보"

    rows = []
    for row in sorted(_SCENARIO_ROWS, key=lambda r: r["rank"]):
        enriched = dict(row)
        enriched["decision_note"] = enriched.get("decision_note") or decision_note(enriched)
        rows.append(enriched)
    return {
        "count": len(_SCENARIO_ROWS),
        "scenarios": rows,
    }


def get_optimization_summary() -> dict:
    """Notebook v5/Phase 8 optimization summary."""
    pareto_rows = [dict(r) for r in _SCENARIO_ROWS if r["key"].startswith("pareto_")]
    phase8_rows = [dict(r) for r in _SCENARIO_ROWS if r["key"].startswith("phase8_")]
    return {
        "methods": [dict(r) for r in _OPTIMIZATION_ROWS],
        "pareto_representatives": pareto_rows,
        "phase8_representatives": phase8_rows,
        "available_lookup_keys": [
            "nsga2_max",
            "pareto_aggressive", "pareto_balanced", "pareto_conservative",
            "phase8_max", "phase8_recovery", "phase8_safe",
        ],
    }


def _vector_from_sigmas(sigmas: dict[str, float]):
    import numpy as np

    simulation_core._ensure_loaded()
    state = simulation_core._state
    base = state["tex_base"]
    lower_better = {"ir_pct", "HR9", "BB9", "WHIP", "babip_against", "era_fip_diff"}

    def sigma_to_delta(feature: str, sigma: float) -> float:
        sign = -1 if feature in lower_better else 1
        return sign * sigma * float(state["feat_std"][feature])

    adj = {**base}
    for feature, sigma in sigmas.items():
        adj[feature] += sigma_to_delta(feature, sigma)
    if any(abs(sigmas.get(f, 0.0)) > 1e-9 for f in ("HR9", "BB9")):
        cfip_base = (
            (base["ERA"] - base["era_fip_diff"])
            - (13 * base["HR9"] - 2 * base["K9"] + 3 * base["BB9"]) / 9
        )
        adj["era_fip_diff"] = adj["ERA"] - (
            cfip_base + (13 * adj["HR9"] - 2 * adj["K9"] + 3 * adj["BB9"]) / 9
        )
    return np.array([[adj[f] for f in simulation_core.MODEL_FEATURES]])


def _predict_raw_matrix(raw):
    import numpy as np

    state = simulation_core._state
    scaled = state["scaler"].transform(raw)
    preds = np.stack(
        [
            state["ridge"].predict(scaled),
            state["lasso"].predict(scaled),
            state["rf"].predict(raw),
            state["xgb"].predict(raw),
        ],
        axis=1,
    )
    return preds.mean(axis=1), preds.std(axis=1)


def load_optimization_datasets() -> dict:
    """Load cached full optimization datasets if present."""
    nsga_path = _OUTPUT_DIR / "optimization_nsga2_front.csv"
    out = {
        "nsga2_path": str(nsga_path),
        "nsga2_front": [],
    }
    if nsga_path.exists():
        out["nsga2_front"] = pd.read_csv(nsga_path).to_dict(orient="records")
    out["has_nsga2"] = bool(out["nsga2_front"])
    return out


def run_nsga2_optimization(pop_size: int = 200, n_gen: int = 150, force: bool = False) -> dict:
    """Run NSGA-II and save the full Pareto front to CSV."""
    path = _OUTPUT_DIR / "optimization_nsga2_front.csv"
    if path.exists() and not force:
        df = pd.read_csv(path)
        return {"path": str(path), "rows": len(df), "front": df.to_dict(orient="records"), "cached": True}

    import numpy as np
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.core.problem import Problem
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.operators.sampling.rnd import FloatRandomSampling
    from pymoo.termination import get_termination

    simulation_core._ensure_loaded()
    state = simulation_core._state
    keys = _optimization_feature_keys()

    class ScenarioProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=len(keys),
                n_obj=2,
                xl=np.full(len(keys), -1.5),
                xu=np.full(len(keys), 1.5),
            )

        def _evaluate(self, x_pop, out, *args, **kwargs):
            raw = np.vstack([
                _vector_from_sigmas(dict(zip(keys, x))).reshape(1, -1)
                for x in x_pop
            ])
            means, stds = _predict_raw_matrix(raw)
            out["F"] = np.column_stack([-(means - state["base_resid"]), stds])

    res = pymoo_minimize(
        ScenarioProblem(),
        NSGA2(
            pop_size=pop_size,
            sampling=FloatRandomSampling(),
            crossover=SBX(prob=0.9, eta=15),
            mutation=PM(eta=20),
            eliminate_duplicates=True,
        ),
        get_termination("n_gen", n_gen),
        seed=42,
        verbose=False,
    )

    deltas = -res.F[:, 0]
    stds = res.F[:, 1]
    rows = []
    for i, x in enumerate(res.X):
        row = {
            "point": i + 1,
            "delta": round(float(deltas[i]), 6),
            "pred_std": round(float(stds[i]), 6),
            "predicted_W": round(float(state["tex_actual_w"] + deltas[i]), 6),
        }
        row.update({f"sigma_{k}": round(float(v), 6) for k, v in zip(keys, x)})
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    _OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return {"path": str(path), "rows": len(df), "front": df.to_dict(orient="records"), "cached": False}

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


# ============================================================
# 7. 시각화 — 시나리오 비교 막대그래프
# ============================================================

def plot_scenario_comparison(scenarios: list[str]) -> dict:
    """여러 시나리오의 예상 승수를 baseline과 막대그래프로 비교.

    이름들은 lookup_pareto와 동일하게 받는다 (Pareto 또는 Grid).
    'baseline'은 자동 포함되며, 막대 위에 delta가 함께 표시된다.

    Args:
        scenarios: 시나리오 이름 리스트.
                   예: ['aggressive', 'balanced', 'conservative']
                   또는 ['best_overall', 'best_pitching', 'baseline']

    Returns:
        plotly Figure spec + 요약 데이터. UI는 'spec'을 그대로 plotly로 그림.
    """
    import plotly.graph_objects as go

    baseline_W = float(_SCENARIO_BY_KEY["baseline"]["predicted_W"])

    labels = ['baseline (현재 수준)']
    values = [baseline_W]
    deltas = [0.0]
    stds = [0.0]
    skipped = []

    for name in scenarios:
        if str(name).strip().lower() == 'baseline':
            continue
        r = lookup_pareto(name)
        if 'error' in r:
            skipped.append({'name': name, 'reason': r['error']})
            continue
        if 'predicted_W' in r:
            w = float(r['predicted_W'])
        elif 'mean_W' in r:
            w = float(r['mean_W'])
        else:
            w = baseline_W + float(r.get('delta', 0))
        labels.append(str(r.get('label', name)))
        values.append(round(w, 2))
        deltas.append(round(float(r.get('delta', w - baseline_W)), 2))
        stds.append(float(r.get('pred_std', 0)))

    if len(values) < 2:
        return {'error': f'유효한 시나리오가 없습니다 ({scenarios})', 'skipped': skipped}

    text = [f'{values[0]}승']
    for i in range(1, len(values)):
        sign = '+' if deltas[i] >= 0 else ''
        text.append(f'{values[i]}승 ({sign}{deltas[i]:.1f})')

    colors = ['#888888'] + ['#003278'] * (len(values) - 1)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        text=text,
        textposition='outside',
        marker_color=colors,
        error_y=dict(type='data', array=stds, visible=True, thickness=1, color='#666'),
        hovertemplate='<b>%{x}</b><br>W: %{y}<br>±σ: %{error_y.array}<extra></extra>',
    ))
    fig.update_layout(
        title='시나리오별 예상 승수 (보정)',
        yaxis_title='승수 (W)',
        showlegend=False,
        height=420,
        margin=dict(t=60, b=60, l=60, r=20),
    )
    fig.add_hline(y=baseline_W, line_dash='dot', line_color='#888',
                  annotation_text=f'baseline {baseline_W}', annotation_position='right')

    out = {
        'type': 'plotly_figure',
        'spec': fig.to_json(),
        'title': '시나리오 W 비교',
        'caption': f'baseline {baseline_W}승 대비 {len(values) - 1}개 시나리오',
        'data': dict(zip(labels, values)),
    }
    if skipped:
        out['skipped'] = skipped
    return out


# ============================================================
# 8. 시각화 — historical 잔차 분포
# ============================================================

def plot_historical_distribution(
    year_from: int | None = None,
    year_to: int | None = None,
    residual_min: float | None = None,
    residual_max: float | None = None,
    highlight_tex_2025: bool = True,
) -> dict:
    """historical 팀-시즌 잔차 히스토그램. TEX 2025(-9.06) 위치를 점선으로 표시.

    "역대 잔차 분포 어떻게 돼?", "TEX 2025 잔차가 얼마나 극단적이야?" 같은
    위치 비교 질문에 적합.

    Args:
        year_from / year_to: 연도 범위.
        residual_min / residual_max: 잔차 범위 필터.
        highlight_tex_2025: TEX 2025 잔차(-9.06) 수직 점선 표시 여부.

    Returns:
        plotly Figure spec + 요약 통계 (n, 평균, std, TEX 2025 percentile).
    """
    import plotly.graph_objects as go

    path = _DATA_DIR / "mlb_team_seasons.csv"
    df = pd.read_csv(path)
    df['pyth_wp'] = df['RS']**1.83 / (df['RS']**1.83 + df['RA']**1.83)
    df['pyth_W'] = (df['pyth_wp'] * df['G']).round(1)
    df['residual'] = df['W'] - df['pyth_W']

    if year_from is not None:
        df = df[df['year'] >= year_from]
    if year_to is not None:
        df = df[df['year'] <= year_to]
    if residual_min is not None:
        df = df[df['residual'] >= residual_min]
    if residual_max is not None:
        df = df[df['residual'] <= residual_max]

    if df.empty:
        return {'error': '조건에 맞는 데이터 없음'}

    tex_2025_residual = -9.06
    pct_below_tex = float((df['residual'] < tex_2025_residual).sum() / len(df) * 100)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df['residual'],
        nbinsx=30,
        marker_color='#003278',
        opacity=0.75,
        name='Historical',
        hovertemplate='잔차 %{x:.1f}<br>%{y}개 시즌<extra></extra>',
    ))
    if highlight_tex_2025:
        fig.add_vline(
            x=tex_2025_residual,
            line_dash='dash',
            line_color='#C0111F',
            line_width=2,
            annotation_text=f'TEX 2025 ({tex_2025_residual})',
            annotation_position='top',
        )

    fig.update_layout(
        title=f'Historical 잔차 분포 ({df["year"].min()}~{df["year"].max()}, n={len(df)})',
        xaxis_title='잔차 (W − pyth_W)',
        yaxis_title='팀-시즌 수',
        showlegend=False,
        height=420,
        margin=dict(t=60, b=60, l=60, r=20),
    )

    return {
        'type': 'plotly_figure',
        'spec': fig.to_json(),
        'title': '잔차 분포',
        'caption': (f'{len(df)}개 팀-시즌, 평균 {df["residual"].mean():+.2f}, '
                    f'std {df["residual"].std():.2f}'),
        'percentile_below_tex_2025': round(pct_below_tex, 1),
        'mean': round(float(df['residual'].mean()), 2),
        'std': round(float(df['residual'].std()), 2),
        'n': int(len(df)),
    }


# ============================================================
# 9. 시각화 — 팀 비교 레이더 차트
# ============================================================

# 레이더 축: (컬럼, 표시명, lower_better 여부)
_RADAR_AXES = [
    ('K/9',    'K/9',    False),
    ('BB/9',   'BB/9',   True),
    ('HR/9',   'HR/9',   True),
    ('BABIP',  'BABIP',  True),
    ('GB%',    'GB%',    False),
    ('Hard%',  'Hard%',  True),
    ('SV',     'SV',     False),
    ('BS',     'BS',     True),
]


def plot_team_radar(teams: list[str]) -> dict:
    """TEX와 1-3개 팀의 투수 지표를 레이더 차트로 비교.

    8개 축 (K/9, BB/9, HR/9, BABIP, GB%, Hard%, SV, BS)을 MLB 30팀 기준으로
    0~1 정규화한다. **bigger polygon = better**가 되도록 lower-better 지표
    (BB/9, HR/9, BABIP, Hard%, BS)는 역방향 처리한다.

    Args:
        teams: 비교 대상 팀 리스트. TEX는 자동 포함되므로 넣지 않아도 됨.
               예: ['SEA'] → TEX vs SEA, ['SEA', 'HOU'] → TEX vs SEA vs HOU.
               코드(SEA/HOU/...) 또는 한국어 별명(매리너스 등) 모두 지원.

    Returns:
        plotly Figure spec + 정규화된 값 표 + 원본 값 표.
    """
    import plotly.graph_objects as go

    path = _DATA_DIR / "mlb_teams_2025_pitching.csv"
    df = pd.read_csv(path)

    # TEX 자동 포함 + 입력 정규화 (중복·누락·미존재 처리)
    codes = ['TEX']
    skipped = []
    for t in teams:
        c = _normalize_team(t)
        if c == 'TEX' or c in codes:
            continue
        if c not in df['Team'].values:
            skipped.append({'name': t, 'reason': f'팀 코드 없음: {c}'})
            continue
        codes.append(c)
        if len(codes) >= 4:  # TEX + 3개로 가독성 제한
            break

    if len(codes) < 2:
        return {
            'error': '비교할 유효 팀이 없습니다.',
            'available_teams': sorted(df['Team'].unique()),
            'skipped': skipped,
        }

    # 30팀 기준 min-max 정규화 + lower-better 반전
    axis_labels = []
    raw_table: dict[str, dict[str, float]] = {c: {} for c in codes}
    norm_table: dict[str, list[float]] = {c: [] for c in codes}
    for col, label, lb in _RADAR_AXES:
        if col not in df.columns:
            continue
        axis_labels.append(label)
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        rng = col_max - col_min if col_max > col_min else 1.0
        for c in codes:
            raw = float(df.loc[df['Team'] == c, col].iloc[0])
            raw_table[c][label] = round(raw, 3)
            normalized = (col_max - raw) / rng if lb else (raw - col_min) / rng
            norm_table[c].append(round(normalized, 3))

    # 폐곡선 — 첫 점을 마지막에 다시 추가
    closed_labels = axis_labels + axis_labels[:1]

    # TEX는 진한 파랑, 비교팀들은 빨강·주황·녹색 순
    palette = ['#003278', '#C0111F', '#FF8C00', '#2E8B57']

    fig = go.Figure()
    for i, c in enumerate(codes):
        values = norm_table[c] + norm_table[c][:1]
        color = palette[i % len(palette)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=closed_labels,
            fill='toself',
            name=c,
            line=dict(color=color, width=2),
            opacity=0.6,
            hovertemplate='<b>%{theta}</b><br>정규화: %{r:.2f}<extra>' + c + '</extra>',
        ))

    fig.update_layout(
        title=f'투수 지표 레이더 — {" vs ".join(codes)}',
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickvals=[0.25, 0.5, 0.75, 1.0]),
        ),
        showlegend=True,
        height=460,
        margin=dict(t=70, b=40, l=40, r=40),
    )

    out = {
        'type': 'plotly_figure',
        'spec': fig.to_json(),
        'title': '팀 비교 레이더',
        'caption': (f'{" vs ".join(codes)} | bigger = better '
                    f'(lower-better 지표는 역방향 처리됨)'),
        'normalized_0to1': {c: dict(zip(axis_labels, norm_table[c])) for c in codes},
        'raw_values': raw_table,
    }
    if skipped:
        out['skipped'] = skipped
    return out


# ============================================================
# 10. 데이터 디스커버리 — list / describe / query
# ============================================================

# (파일명 stem) → 한 줄 설명. LLM에게 데이터셋 카탈로그 역할.
DATA_CATALOG: dict[str, str] = {
    # 팀/리그 기준 (multi-team / multi-year)
    'mlb_team_seasons': '10개년 MLB 30팀 시즌 통계 (W/L/RS/RA/홈어웨이wp) — historical 잔차 비교용',
    'mlb_teams_2025_pitching': '2025 MLB 30팀 투수 시즌 집계 (ERA/FIP/K9/BB9/HR9/BABIP/SV/BS/GB%/Hard%)',
    'mlb_2025_batting_order_splits': '2025 MLB 30팀 타순별(1~9번) wOBA',
    'batting_stats_2025_all': '2025 MLB 전체 타자 시즌 통계',
    'sprint_speed_2025': '2025 MLB 모든 선수 주루 sprint speed',
    'hit_by_pitch': '2025 MLB 사구 데이터',
    'fangraphs-guts-data': 'FanGraphs Guts (선형무기 run value 가이드)',

    # TEX 투수 (2025)
    'rangers_2025_pitchers_daily_final': '2025 TEX 투수 일별 Statcast (구속/구종/whiff/release)',
    'rangers_pitcher_gamelogs': '2025 TEX 투수 경기별 게임로그 (IP/ER/K/BB/SVO/홀드)',
    'rangers_pitching_statcast': '2025 TEX 투수 시즌 Statcast 집계',
    'rangers_pitching_batted': '2025 TEX 투수 타구 유형 분포 (GB%/FB%/LD%)',
    'rangers_pitching_discipline': '2025 TEX 투수 디시플린 (Chase%/Whiff%/CSW%)',
    'texas_pitchers_2025': '2025 TEX 투수 FanGraphs 시즌 집계',
    'texas_2025_pitchers_vs_right': '2025 TEX 투수 vs 우타 split',
    'texas_2025_pitches_vs_left': '2025 TEX 투수 vs 좌타 split',
    'tex_2025_pitching_inning_splits': '2025 TEX 이닝별(1~9) 실점 집계',
    'tex_2025_save_situation_splits': '2025 TEX 세이브 상황별 ERA/WHIP',
    'tex_clutch_pit': '2025 TEX 투수 클러치 지표 (WPA/RE24/Clutch)',

    # TEX 타자 (2025)
    'rangers_2025_batters_daily_final': '2025 TEX 타자 일별 Statcast (EV/LA/Barrel/Whiff/Bat_Speed)',
    'rangers_batter_gamelogs': '2025 TEX 타자 경기별 게임로그',
    'rangers_hitting_statcast': '2025 TEX 타자 시즌 Statcast 집계',
    'rangers_hitting_batted': '2025 TEX 타자 타구 유형 분포',
    'rangers_hitting_discipline': '2025 TEX 타자 디시플린',
    'rangers_running': '2025 TEX 주루 (sprint speed / 도루)',
    'tex_2025_batting_order_splits': '2025 TEX 타순별(1~9) wOBA',
    'tex_clutch_bat': '2025 TEX 타자 클러치 지표',
    'tex_wrc+': '2025 TEX 타자 wRC+',

    # TEX 경기/이벤트
    'rangers_roster_2025': '2025 TEX 로스터 (선수 메타 + 포지션)',
    'texas_2025_game_log': '2025 TEX 경기별 결과 (스코어/홈어웨이/상대)',
    'statcast_tex_all_events_2025': '2025 TEX 모든 타석 이벤트 (raw, 6058행)',
    'statcast_tex_doubles_2025': '2025 TEX 2루타 이벤트만',
    'statcast_tex_singles_2025': '2025 TEX 1루타 이벤트만',

    # 모델 결과 (Final/data/)
    'pitcher_stats_ag': '동작분석 5인 투수 집계 (BABIP 운/실력 분류)',
    'pitcher_stats_mb': '동작분석 5인 투수 메커니즘 분석 결과',
    'model_comparison': '4모델 (Ridge/Lasso/RF/XGB) 잔차 모델 비교',
    'model_summary': 'ML 잔차 모델 학습 요약 (피처 중요도 등)',
}


def _resolve_csv_path(name: str) -> Path | None:
    """카탈로그 이름 → 실제 csv 경로. data_raw 우선, 없으면 data."""
    if name not in DATA_CATALOG:
        return None
    for base in [_DATA_DIR, _APP_ROOT / "data"]:
        p = base / f"{name}.csv"
        if p.exists():
            return p
    return None


def list_data_sources() -> dict:
    """사용 가능한 모든 데이터셋의 이름 + 한 줄 설명 + 행/컬럼 수.

    "어떤 데이터가 있는지", "TEX 클러치 데이터 있어?" 같은 디스커버리에 사용.
    이 결과를 보고 적절한 source를 골라 describe_data/query_data로 진행한다.

    Returns:
        sources: [{'name', 'description', 'rows', 'cols', 'available'}, ...]
        total: 카탈로그 등록된 데이터셋 총 개수
    """
    out = []
    for name, desc in DATA_CATALOG.items():
        path = _resolve_csv_path(name)
        if path is None:
            out.append({'name': name, 'description': desc, 'available': False})
            continue
        try:
            df = pd.read_csv(path, nrows=0)  # 헤더만
            n_rows = sum(1 for _ in open(path, 'r', encoding='utf-8')) - 1
            out.append({
                'name': name,
                'description': desc,
                'rows': n_rows,
                'cols': len(df.columns),
                'available': True,
            })
        except Exception as e:
            out.append({'name': name, 'description': desc, 'available': False, 'error': str(e)})
    return {'total': len(DATA_CATALOG), 'sources': out}


def describe_data(source: str) -> dict:
    """특정 데이터셋의 컬럼 정보 + 샘플 5행. query_data 호출 전 컬럼 확인용.

    Args:
        source: DATA_CATALOG의 키 (예: 'mlb_team_seasons', 'rangers_pitcher_gamelogs').

    Returns:
        name, description, shape, columns(이름/dtype/null수/unique수/min/max), sample(5행).
    """
    if source not in DATA_CATALOG:
        return {
            'error': f'알 수 없는 데이터셋: {source}',
            'hint': 'list_data_sources()로 카탈로그를 먼저 확인하세요.',
        }
    path = _resolve_csv_path(source)
    if path is None:
        return {'error': f'파일이 존재하지 않음: {source}.csv'}

    df = pd.read_csv(path)
    columns = []
    for col in df.columns:
        col_info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'null': int(df[col].isnull().sum()),
            'unique': int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            col_info['min'] = round(float(df[col].min()), 3) if df[col].notna().any() else None
            col_info['max'] = round(float(df[col].max()), 3) if df[col].notna().any() else None
        columns.append(col_info)

    return {
        'name': source,
        'description': DATA_CATALOG[source],
        'shape': {'rows': int(len(df)), 'cols': int(len(df.columns))},
        'columns': columns,
        'sample': df.head(5).to_dict(orient='records'),
    }


def query_data(
    source: str,
    filter: dict | None = None,
    columns: list[str] | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int = 50,
) -> dict:
    """csv를 필터·선택·정렬해서 row 리스트로 반환. 일반 쿼리 도구.

    "9월 1점차 경기에서 가장 많이 등판한 투수", "ERA 5 이상 투수의 GB%" 같은
    조합 질문을 처리할 때 사용. 기존 query_gamelog/query_team_history의 일반화.

    필터 문법 (dict):
        - 등호:        {'month': 9, 'opponent': 'SEA'}
        - 범위:        {'ERA': {'gte': 4.0, 'lte': 6.0}}
        - 포함:        {'name': {'contains': 'Garcia'}}
        - 다중 값 OR:  {'pos': {'in': ['SP', 'RP']}}
        AND로만 결합 (모든 조건 동시 만족).

    Args:
        source: DATA_CATALOG의 키.
        filter: 위 문법 dict. None이면 전체.
        columns: 반환할 컬럼만. None이면 전체.
        sort_by: 정렬 컬럼.
        ascending: 오름차순 여부.
        limit: 반환 행 수 상한 (기본 50, 최대 200).

    Returns:
        matched(필터 결과 행수), returned(실제 반환 행수), rows(dict 리스트), columns_used.
    """
    if source not in DATA_CATALOG:
        return {'error': f'알 수 없는 데이터셋: {source}', 'hint': 'list_data_sources() 확인.'}
    path = _resolve_csv_path(source)
    if path is None:
        return {'error': f'파일이 존재하지 않음: {source}.csv'}

    df = pd.read_csv(path)
    mask = pd.Series(True, index=df.index)
    filter_errors = []

    if filter:
        for col, cond in filter.items():
            if col not in df.columns:
                filter_errors.append(f'존재하지 않는 컬럼: {col}')
                continue
            if isinstance(cond, dict):
                if 'gte' in cond:
                    mask &= df[col] >= cond['gte']
                if 'gt' in cond:
                    mask &= df[col] > cond['gt']
                if 'lte' in cond:
                    mask &= df[col] <= cond['lte']
                if 'lt' in cond:
                    mask &= df[col] < cond['lt']
                if 'in' in cond:
                    mask &= df[col].isin(cond['in'])
                if 'contains' in cond:
                    mask &= df[col].astype(str).str.contains(str(cond['contains']), case=False, na=False)
            else:
                mask &= df[col] == cond

    sub = df[mask]
    if sort_by:
        if sort_by not in df.columns:
            filter_errors.append(f'정렬 컬럼 없음: {sort_by}')
        else:
            sub = sub.sort_values(sort_by, ascending=ascending)

    limit = max(1, min(int(limit), 200))
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            filter_errors.append(f'존재하지 않는 컬럼: {missing}')
        valid_cols = [c for c in columns if c in df.columns]
        result_df = sub[valid_cols].head(limit) if valid_cols else sub.head(limit)
    else:
        result_df = sub.head(limit)

    out = {
        'source': source,
        'matched': int(len(sub)),
        'returned': int(len(result_df)),
        'rows': result_df.to_dict(orient='records'),
        'columns_used': list(result_df.columns),
    }
    if filter_errors:
        out['warnings'] = filter_errors
    return out


# ============================================================
# 11. 시각화 — 임의 데이터셋·컬럼 차트 (plot_custom)
# ============================================================

_PLOT_TYPES = {'bar', 'line', 'scatter', 'histogram', 'box'}


def plot_custom(
    source: str,
    chart_type: str,
    x: str,
    y: str | list[str] | None = None,
    color_by: str | None = None,
    filter: dict | None = None,
    sort_by: str | None = None,
    ascending: bool = True,
    limit: int = 500,
    title: str | None = None,
) -> dict:
    """DATA_CATALOG의 임의 csv에서 임의 컬럼 조합으로 plotly 차트 생성.

    전용 plot_* 도구로 못 만드는 시각화를 LLM이 자유롭게 조합하여 만들 때 사용.
    "월별 TEX 승률 라인", "투수별 K9 vs ERA 산점도" 같은 ad-hoc 시각화 처리.

    Args:
        source: DATA_CATALOG의 키 (list_data_sources / describe_data로 확인 가능).
        chart_type: 'bar' | 'line' | 'scatter' | 'histogram' | 'box'.
        x: x축 컬럼.
        y: y축 컬럼 (histogram은 무시). 다중 시리즈는 list로 (line/bar에서).
        color_by: 색상 그룹핑 컬럼. None이면 단색.
        filter: query_data와 동일 문법
                ({'col': value}, {'col': {'gte': 4.0, 'lte': 6.0}}, {'col': {'in': [...]}}, ...)
        sort_by: 정렬 컬럼 (line에서 권장 — x축 정렬 효과).
        ascending: 오름차순 여부.
        limit: 그릴 행 수 상한 (기본 500, 최대 5000). 대용량 csv 안전.
        title: 차트 제목. None이면 자동 생성.

    Returns:
        plotly Figure spec + 사용 행수 + 적용된 필터 메타.
    """
    import plotly.express as px

    if source not in DATA_CATALOG:
        return {'error': f'알 수 없는 데이터셋: {source}', 'hint': 'list_data_sources() 확인.'}
    path = _resolve_csv_path(source)
    if path is None:
        return {'error': f'파일이 존재하지 않음: {source}.csv'}

    chart_type = chart_type.lower().strip()
    if chart_type not in _PLOT_TYPES:
        return {
            'error': f'지원하지 않는 chart_type: {chart_type}',
            'valid_types': sorted(_PLOT_TYPES),
        }

    df = pd.read_csv(path)
    warnings_ = []

    # 필터 적용 (query_data와 동일 문법)
    if filter:
        mask = pd.Series(True, index=df.index)
        for col, cond in filter.items():
            if col not in df.columns:
                warnings_.append(f'존재하지 않는 컬럼 (필터): {col}')
                continue
            if isinstance(cond, dict):
                if 'gte' in cond:
                    mask &= df[col] >= cond['gte']
                if 'gt' in cond:
                    mask &= df[col] > cond['gt']
                if 'lte' in cond:
                    mask &= df[col] <= cond['lte']
                if 'lt' in cond:
                    mask &= df[col] < cond['lt']
                if 'in' in cond:
                    mask &= df[col].isin(cond['in'])
                if 'contains' in cond:
                    mask &= df[col].astype(str).str.contains(str(cond['contains']), case=False, na=False)
            else:
                mask &= df[col] == cond
        df = df[mask]

    if df.empty:
        return {'error': '필터 후 데이터가 없습니다.', 'warnings': warnings_}

    # 정렬 + 행 제한
    if sort_by:
        if sort_by in df.columns:
            df = df.sort_values(sort_by, ascending=ascending)
        else:
            warnings_.append(f'정렬 컬럼 없음: {sort_by}')
    limit = max(1, min(int(limit), 5000))
    df = df.head(limit)

    # x/y/color_by 컬럼 검증
    if x not in df.columns:
        return {'error': f'x 컬럼 없음: {x}', 'available_columns': list(df.columns)}
    y_used: list[str] | str | None = None
    if y is not None and chart_type != 'histogram':
        if isinstance(y, str):
            if y not in df.columns:
                return {'error': f'y 컬럼 없음: {y}', 'available_columns': list(df.columns)}
            y_used = y
        else:
            missing = [c for c in y if c not in df.columns]
            if missing:
                return {'error': f'y 컬럼 없음: {missing}', 'available_columns': list(df.columns)}
            y_used = list(y)
    if color_by and color_by not in df.columns:
        warnings_.append(f'color_by 컬럼 없음, 무시: {color_by}')
        color_by = None

    # 자동 제목
    if title is None:
        y_str = y_used if isinstance(y_used, str) else (', '.join(y_used) if y_used else '')
        title = f'{source} — {chart_type}({x}{(", " + y_str) if y_str else ""})'

    # 차트 생성
    common_kwargs = {'color': color_by} if color_by else {}
    try:
        if chart_type == 'histogram':
            fig = px.histogram(df, x=x, **common_kwargs)
        elif chart_type == 'bar':
            fig = px.bar(df, x=x, y=y_used, **common_kwargs)
        elif chart_type == 'line':
            fig = px.line(df, x=x, y=y_used, **common_kwargs)
        elif chart_type == 'scatter':
            fig = px.scatter(df, x=x, y=y_used, **common_kwargs)
        elif chart_type == 'box':
            fig = px.box(df, x=x, y=y_used, **common_kwargs)
        else:
            return {'error': f'예외: {chart_type}'}
    except Exception as e:
        return {'error': f'차트 생성 실패: {e}', 'hint': '컬럼 dtype 확인 — bar/line/scatter는 y가 숫자형이어야 합니다.'}

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(t=60, b=60, l=60, r=20),
    )

    out = {
        'type': 'plotly_figure',
        'spec': fig.to_json(),
        'title': title,
        'caption': f'{source} | {chart_type} | {len(df)}행 사용' + (f' | color={color_by}' if color_by else ''),
        'rows_used': int(len(df)),
        'columns_used': {
            'x': x,
            'y': y_used,
            'color_by': color_by,
        },
    }
    if filter:
        out['filter_applied'] = filter
    if warnings_:
        out['warnings'] = warnings_
    return out


# ============================================================
# 12. 선수 성적 조회
# ============================================================

def get_player_stats(name: str) -> dict:
    """선수 이름으로 TEX 2025 주요 성적을 한 번에 조회.

    투수·타자 구분 없이 이름(부분 일치 가능)으로 검색하며,
    관련 CSV 여러 개를 합쳐 핵심 지표를 반환한다.

    투수 결과: ERA / FIP / K/9 / BB/9 / HR/9 / BABIP / GB% / WAR / SV / BS
              + 클러치 지표(WPA / Clutch / pLI)
              + 경기별 게임로그 요약(경기수 / 평균 IP / 평균 피실점)
    타자 결과: wRC+ / wOBA / xwOBA / EV / Barrel% / HR / BB
              + 클러치 지표(WPA / Clutch)

    Args:
        name: 선수 이름 또는 성 (예: 'Garcia', 'Seager', 'Leiter').
              대소문자 무관, 부분 일치 허용.

    Returns:
        found(bool), player_type('pitcher'|'batter'|'both'|'not_found'),
        pitcher(투수 지표 dict), batter(타자 지표 dict), note(메시지).
    """
    name_q = name.strip().lower()
    result: dict = {'query': name, 'found': False, 'pitcher': None, 'batter': None}

    # ── 투수 ───────────────────────────────────────────────
    pitcher_rows: list[dict] = []

    # FanGraphs 시즌 집계
    p_fg_path = _resolve_csv_path('texas_pitchers_2025')
    if p_fg_path:
        df_fg = pd.read_csv(p_fg_path)
        match = df_fg[df_fg['Name'].str.lower().str.contains(name_q, na=False)]
        for _, row in match.iterrows():
            sv, bs = int(row.get('SV', 0) or 0), int(row.get('BS', 0) or 0)
            role = 'SP' if int(row.get('GS', 0) or 0) >= 5 else 'RP'
            pitcher_rows.append({
                'name': row['Name'],
                'role': role,
                'G': int(row.get('G', 0) or 0),
                'GS': int(row.get('GS', 0) or 0),
                'IP': round(float(row.get('IP', 0) or 0), 1),
                'ERA': round(float(row.get('ERA', 0) or 0), 3),
                'FIP': round(float(row.get('FIP', 0) or 0), 3),
                'xFIP': round(float(row.get('xFIP', 0) or 0), 3),
                'K/9': round(float(row.get('K/9', 0) or 0), 2),
                'BB/9': round(float(row.get('BB/9', 0) or 0), 2),
                'HR/9': round(float(row.get('HR/9', 0) or 0), 2),
                'BABIP': round(float(row.get('BABIP', 0) or 0), 3),
                'GB%': round(float(row.get('GB%', 0) or 0), 3),
                'WHIP': round(float(row.get('WHIP', 0) or 0), 3),
                'WAR': round(float(row.get('WAR', 0) or 0), 2),
                'SV': sv,
                'BS': bs,
                'SV_opp': sv + bs,
                'sv_pct': round(sv / (sv + bs), 3) if (sv + bs) > 0 else None,
                'source': 'FanGraphs 시즌 집계',
            })

    # 클러치 지표
    p_cl_path = _resolve_csv_path('tex_clutch_pit')
    if p_cl_path and pitcher_rows:
        df_cl = pd.read_csv(p_cl_path)
        for row_d in pitcher_rows:
            match_cl = df_cl[df_cl['Name'].str.lower().str.contains(name_q, na=False)]
            if not match_cl.empty:
                r = match_cl.iloc[0]
                row_d['WPA'] = round(float(r.get('WPA', 0) or 0), 3)
                row_d['Clutch'] = round(float(r.get('Clutch', 0) or 0), 3)
                row_d['pLI'] = round(float(r.get('pLI', 0) or 0), 3)

    # 게임로그 요약
    p_gl_path = _resolve_csv_path('rangers_pitcher_gamelogs')
    if p_gl_path and pitcher_rows:
        df_gl = pd.read_csv(p_gl_path)
        match_gl = df_gl[df_gl['name'].str.lower().str.contains(name_q, na=False)]
        if not match_gl.empty:
            def _ip(s):
                try:
                    p = str(s).split('.')
                    return int(p[0]) + int(p[1]) / 3 if len(p) > 1 else int(p[0])
                except Exception:
                    return 0.0
            ip_vals = match_gl['IP'].apply(_ip)
            for row_d in pitcher_rows:
                row_d['gamelog_games'] = int(len(match_gl))
                row_d['gamelog_avg_IP'] = round(float(ip_vals.mean()), 2) if len(ip_vals) else None
                row_d['gamelog_avg_ER'] = round(float(match_gl['ER'].mean()), 2) if 'ER' in match_gl.columns else None

    # ── 타자 ───────────────────────────────────────────────
    batter_rows: list[dict] = []

    # wRC+
    wrc_path = _resolve_csv_path('tex_wrc+')
    if wrc_path:
        df_wrc = pd.read_csv(wrc_path)
        match_b = df_wrc[df_wrc['Name'].str.lower().str.contains(name_q, na=False)]
        for _, row in match_b.iterrows():
            batter_rows.append({
                'name': row['Name'],
                'wRC+': round(float(row.get('wRC+', 0) or 0), 1),
                'source': 'FanGraphs wRC+',
            })

    # Statcast 지표
    sc_path = _resolve_csv_path('rangers_hitting_statcast')
    if sc_path:
        df_sc = pd.read_csv(sc_path)
        match_sc = df_sc[df_sc['Player'].str.lower().str.contains(name_q, na=False)]
        for _, row in match_sc.iterrows():
            existing = next((r for r in batter_rows if name_q in r['name'].lower()), None)
            sc_data = {
                'PA': int(row.get('PA', 0) or 0),
                'HR': int(row.get('HR', 0) or 0),
                'BB': int(row.get('BB', 0) or 0),
                'wOBA': round(float(row.get('wOBA', 0) or 0), 3),
                'xwOBA': round(float(row.get('xwOBA', 0) or 0), 3),
                'Exit Velocity': round(float(row.get('Exit Velocity', 0) or 0), 1),
                'Barrel%': round(float(row.get('Barrel %', 0) or 0), 1),
                'Hard Hit%': round(float(row.get('Hard Hit %', 0) or 0), 1),
                'xBA': round(float(row.get('xBA', 0) or 0), 3),
                'xSLG': round(float(row.get('xSLG', 0) or 0), 3),
            }
            if existing:
                existing.update(sc_data)
            else:
                sc_data['name'] = row['Player']
                sc_data['source'] = 'Statcast'
                batter_rows.append(sc_data)

    # 클러치 지표 (타자)
    b_cl_path = _resolve_csv_path('tex_clutch_bat')
    if b_cl_path and batter_rows:
        df_bcl = pd.read_csv(b_cl_path)
        match_bcl = df_bcl[df_bcl['Name'].str.lower().str.contains(name_q, na=False)]
        if not match_bcl.empty:
            r = match_bcl.iloc[0]
            for row_d in batter_rows:
                row_d['WPA'] = round(float(r.get('WPA', 0) or 0), 3)
                row_d['Clutch'] = round(float(r.get('Clutch', 0) or 0), 3)
                row_d['pLI'] = round(float(r.get('pLI', 0) or 0), 3)

    # ── 결과 조합 ───────────────────────────────────────────
    found = bool(pitcher_rows or batter_rows)
    if pitcher_rows and batter_rows:
        player_type = 'both'
    elif pitcher_rows:
        player_type = 'pitcher'
    elif batter_rows:
        player_type = 'batter'
    else:
        player_type = 'not_found'

    result['found'] = found
    result['player_type'] = player_type
    if pitcher_rows:
        result['pitcher'] = pitcher_rows[0] if len(pitcher_rows) == 1 else pitcher_rows
    if batter_rows:
        result['batter'] = batter_rows[0] if len(batter_rows) == 1 else batter_rows
    if not found:
        result['note'] = (
            f"'{name}'과 일치하는 TEX 2025 선수를 찾지 못했습니다. "
            "성(last name)이나 영문 전체 이름으로 다시 시도하세요."
        )
    return result


# ============================================================
# 13. 시뮬레이션 → 선수별 기여 분석
# ============================================================

# 시뮬 피처 → texas_pitchers_2025 컬럼 매핑
_SIM_TO_PITCHER_COL: dict[str, str] = {
    'K9':           'K/9',
    'BB9':          'BB/9',
    'HR9':          'HR/9',
    'babip_against':'BABIP',
    'WHIP':         'WHIP',
    'go_ao':        'GB%',
}


def simulation_player_breakdown(sigmas: dict[str, float]) -> dict:
    """시뮬레이션 σ 조정이 실제 선수 성적과 어떻게 연결되는지 분석.

    estimate_residual_scenario와 동일한 sigmas dict를 받아서,
    각 피처 개선이 TEX 투수진 중 '누구를 얼마나 바꿔야 달성 가능한지'를
    선수별로 구체화한다.

    매핑 가능한 피처: K9, BB9, HR9, babip_against, WHIP, go_ao.
    sv_pct / onerun_wp / xi_wp 등 팀 집계 피처는 별도 설명.

    Args:
        sigmas: {'K9': 0.3, 'BB9': 0.4} 형태. estimate_residual_scenario와 동일.
                양수 = 개선 방향.

    Returns:
        - sim_result: estimate_residual_scenario 결과 (예상 승수 + delta)
        - breakdowns: 피처별 {feature, col, tex_avg, target_value, sigma,
                       delta_raw, players(이름/현재값/목표값/gap/부담도)} 리스트
        - team_features: 선수 단위 매핑이 없는 팀 집계 피처 설명
        - summary: 핵심 해석 문장
    """
    from . import simulation_core

    # 1. 시뮬 결과
    sim_result = simulation_core.estimate_residual_scenario(sigmas)
    feat_info = simulation_core.feature_info()
    feat_std = feat_info['feature_std']
    lower_better = set(feat_info['lower_better'])

    # 2. 투수 데이터 로드
    p_path = _resolve_csv_path('texas_pitchers_2025')
    if p_path is None:
        return {'error': 'texas_pitchers_2025.csv 없음', 'sim_result': sim_result}
    df_p = pd.read_csv(p_path)
    df_p['IP_f'] = pd.to_numeric(df_p['IP'], errors='coerce').fillna(0)

    # TEX 팀 IP 가중 평균 계산
    def _wavg(col: str) -> float:
        vals = pd.to_numeric(df_p[col], errors='coerce')
        w = df_p['IP_f']
        valid = vals.notna() & (w > 0)
        if not valid.any():
            return float(vals.mean())
        return float((vals[valid] * w[valid]).sum() / w[valid].sum())

    # 3. 피처별 분석
    breakdowns = []
    team_features = []

    for feat, sigma in sigmas.items():
        if feat not in _SIM_TO_PITCHER_COL:
            # 팀 집계 피처 설명
            desc_map = {
                'sv_pct':        '세이브 성공률 (SV / SV+BS) — 불펜 마무리 운영 개선으로 달성',
                'SV_pg':         '9이닝당 세이브 수 — 선발 완투·구원 투수 운영 전략',
                'onerun_wp':     '1점차 경기 승률 — 클러치 상황 불펜·타선 집중도',
                'xi_wp':         '득실 동률 경기 승률 — 연장·접전 결정력',
                'home_away_diff':'홈/원정 승률 차이 — 구장 활용도',
                'ir_pct':        '물려받은 주자 실점률 — 중계 투수 화력 차단 능력',
                'era_fip_diff':  'ERA - FIP 차이 — BABIP 운 요소 제거 시 실제 투구력 수렴',
                'sb_pct':        '도루 성공률 — 포수 송구·배터리 케어',
                'k_bb':          'K/BB 비율 — 삼진/볼넷 복합 지표 (K9·BB9 조합 시 중복)',
            }
            team_features.append({
                'feature': feat,
                'sigma': sigma,
                'delta_raw': round(sigma * feat_std.get(feat, 0), 4),
                'description': desc_map.get(feat, '팀 집계 지표 — 선수 개인 매핑 불가'),
            })
            continue

        col = _SIM_TO_PITCHER_COL[feat]
        if col not in df_p.columns:
            continue

        tex_avg = _wavg(col)
        std = feat_std.get(feat, 0)
        delta_raw = sigma * std
        # lower_better면 개선 = 감소
        if feat in lower_better:
            target = tex_avg - delta_raw
        else:
            target = tex_avg + delta_raw

        # 선수별 gap 계산
        players_out = []
        for _, row in df_p.sort_values('IP_f', ascending=False).head(15).iterrows():
            val = pd.to_numeric(row.get(col), errors='coerce')
            if pd.isna(val):
                continue
            ip = float(row.get('IP_f', 0))
            if feat in lower_better:
                gap = val - target  # 양수 = 아직 개선 여지 있음
            else:
                gap = target - val  # 양수 = 아직 부족
            ip_total = df_p['IP_f'].sum()
            ip_share = round(ip / ip_total, 3) if ip_total > 0 else 0

            players_out.append({
                'name': row['Name'],
                'role': 'SP' if int(row.get('GS', 0) or 0) >= 5 else 'RP',
                'IP': round(ip, 1),
                'IP_share': ip_share,
                'current': round(float(val), 3),
                'target': round(float(target), 3),
                'gap': round(float(gap), 3),
                'needs_improvement': bool(gap > 0),
            })

        # gap 큰 순 정렬 (개선 여지 많은 선수 우선)
        players_out.sort(key=lambda x: x['gap'], reverse=True)

        breakdowns.append({
            'feature': feat,
            'col': col,
            'sigma': sigma,
            'delta_raw': round(delta_raw, 4),
            'lower_better': feat in lower_better,
            'tex_team_avg': round(tex_avg, 3),
            'target_value': round(target, 3),
            'players': players_out[:8],
        })

    # 4. 요약 문장 생성
    delta = sim_result.get('delta', 0)
    pred_w = sim_result.get('predicted_W_calibrated', 81)
    summary_parts = [f"예상 승수 {pred_w:.1f}승 (베이스라인 대비 +{delta:.2f}승)"]
    for bd in breakdowns:
        needs = [p['name'].split()[-1] for p in bd['players'] if p['needs_improvement']][:3]
        if needs:
            summary_parts.append(
                f"{bd['col']} 목표 {bd['target_value']:.2f}: "
                f"{', '.join(needs)} 등 개선 필요"
            )

    return {
        'sim_result': {
            'predicted_W_calibrated': pred_w,
            'delta': round(delta, 3),
            'pred_std': sim_result.get('pred_std'),
        },
        'breakdowns': breakdowns,
        'team_features': team_features,
        'summary': ' | '.join(summary_parts),
    }
