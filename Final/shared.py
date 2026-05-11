"""Shared utilities, constants, and data for TEX 2025 dashboard."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path
import json
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import base64
from textwrap import wrap as textwrap_wrap

from simulator import (
    build_scenario_snapshots,
    get_batter_options as _get_batter_options,
    get_scenario_defaults as _get_scenario_defaults,
    run_simulation,
)

# ── Colors ────────────────────────────────────────────────────
TEX_BLUE = "#003278"
TEX_NAVY = "#0B1F3A"
TEX_RED  = "#C0111F"
TEX_LIGHT = "#F6F8FB"
TEX_MUTED = "#64748B"

# ── Paths ─────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR  = BASE_DIR / "data_raw"
ASSETS   = BASE_DIR / "assets"

# ── Simulation config ─────────────────────────────────────────
SIMULATION_OPTIONS = [
    "Baseline 2025",
    "Bullpen Upgrade",
    "Hitter Boost",
]
DEFAULT_SCENARIO = "Baseline 2025"
DEFAULT_SIM_RUNS = 200


# ── UI helpers ────────────────────────────────────────────────
def section_badges(*items):
    html = ""
    for text, tone in items:
        cls = "pill"
        if tone == "red":
            cls += " pill-red"
        elif tone == "navy":
            cls += " pill-navy"
        elif tone == "white":
            cls += " pill-white"
        html += f'<span class="{cls}">{text}</span>'
    st.markdown(html, unsafe_allow_html=True)


def finding_box(title, body):
    st.markdown(f"""
    <div class="finding-box">
        <strong>{title}</strong><br>
        {body}
    </div>
    """, unsafe_allow_html=True)


def glass_note(body):
    st.markdown(f'<div class="glass-card">{body}</div>', unsafe_allow_html=True)


def glossary_box(title: str, terms: dict[str, str]):
    if "?" in title or "吏" in title:
        title = "키네마틱 지표 용어"
    rows = "".join(
        f'<div style="display:grid;grid-template-columns:minmax(88px,0.28fr) 1fr;gap:10px;'
        f'padding:8px 0;border-top:1px solid rgba(13,27,51,0.08);">'
        f'<div style="font-weight:800;color:#0D1B33;">{term}</div>'
        f'<div style="color:#475569;">{desc}</div></div>'
        for term, desc in terms.items()
    )
    st.markdown(f"""
    <div class="glass-card">
        <div class="chart-title">{title}</div>
        <div class="chart-caption" style="margin-bottom:18px; padding-bottom:2px;">
            표와 그래프를 읽기 전에 필요한 용어만 짧게 정리했습니다.
        </div>
        <div style="font-size:13px;line-height:1.55;margin-top:10px;">{rows}</div>
    </div>
    """, unsafe_allow_html=True)


KINEMATIC_TERMS = {
    "HSS": "Hip-Shoulder Separation. 골반과 어깨가 얼마나 분리되어 회전하는지 보는 지표입니다. 투구 시 에너지 전달과 관련이 있습니다.",
    "HSS @ FP": "Foot Plant, 즉 앞발이 땅에 닿는 순간의 HSS입니다. 릴리스 전 몸의 꼬임이 얼마나 만들어졌는지 봅니다.",
    "HSS max": "투구 동작 전체에서 HSS가 가장 크게 나온 값입니다.",
    "Hip peak 3D": "골반 회전 속도의 최고값입니다. 단위가 °/s이면 1초당 몇 도 회전했는지를 뜻합니다.",
    "Trunk peak 3D": "몸통 회전 속도의 최고값입니다. 골반 회전 뒤 몸통이 얼마나 빠르게 따라오는지 봅니다.",
    "Trunk/Hip ratio": "몸통 최고 회전 속도를 골반 최고 회전 속도로 나눈 값입니다. 몸통 회전이 골반 대비 얼마나 강한지 보여줍니다.",
    "Timing diff": "골반 회전 피크와 몸통 회전 피크 사이의 시간 차이입니다. 순차적인 회전 연결이 잘 되는지 확인합니다.",
}


BASEBALL_TERMS = {
    "K/9": "9이닝당 탈삼진 수입니다. 높을수록 삼진을 많이 잡는 투수입니다.",
    "BB/9": "9이닝당 볼넷 수입니다. 낮을수록 제구가 안정적입니다.",
    "ERA": "평균자책점입니다. 투수가 허용한 자책점을 9이닝 기준으로 환산한 값입니다.",
    "FIP": "수비 영향을 줄이고 삼진, 볼넷, 피홈런 중심으로 투수 성과를 본 지표입니다.",
    "WHIP": "이닝당 허용한 안타와 볼넷 수입니다. 낮을수록 주자를 덜 내보냅니다.",
    "WAR": "대체 선수 대비 승리 기여도입니다. 높을수록 팀 승리에 더 많이 기여했다는 뜻입니다.",
    "세이브(SV) / 블론 세이브(BS)": "마무리 투수의 결과를 보는 두 지표입니다. 세이브는 리드를 지켜 경기를 마무리한 경우, 블론 세이브는 이어받은 리드를 놓친 경우입니다. 블론 세이브가 많을수록 접전에서 팀이 실점에 취약하다는 신호입니다.",
    "Cohen's d": "두 상황의 평균 차이가 얼마나 큰지 보는 효과 크기입니다. 보통 |d|가 클수록 차이가 큽니다.",
    "p-value": "관찰된 차이가 우연일 가능성을 보는 값입니다. 작을수록 통계적으로 차이가 있다고 해석하기 쉽습니다.",
}


def page_hero(kicker: str, title: str, body: str, badges: list | None = None):
    badge_html = ""
    if badges:
        for text, tone in badges:
            cls = "pill pill-white" if tone == "white" else "pill"
            if tone == "red":
                cls = "pill pill-red"
            elif tone == "navy":
                cls = "pill pill-navy"
            badge_html += f'<span class="{cls}">{text}</span>'
    st.markdown(f"""
    <div class="hero-card">
        <div class="page-kicker">{kicker}</div>
        <div style="margin-top:10px;">{badge_html}</div>
        <h1>{title}</h1>
        <p>{body}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, body: str = ""):
    st.markdown(f"""
    <div class="section-shell compact">
        <div class="section-heading">{title}</div>
        <div class="section-copy">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub="", accent="navy"):
    st.markdown(f"""
    <div class="kpi-card-custom accent-{accent}">
        <div class="kpi-label-custom">{label}</div>
        <div class="kpi-value-custom">{value}</div>
        <div class="kpi-sub-custom">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ── Simulation cache ──────────────────────────────────────────

def _build_schedule_context(raw_dir: str) -> pd.DataFrame:
    """texas_2025_game_log.csv → 월별 경기 수/홈원정/실제승률 요약."""
    path = Path(raw_dir) / "texas_2025_game_log.csv"
    if not path.exists():
        return pd.DataFrame()
    gl = pd.read_csv(path)
    gl['Date'] = pd.to_datetime(gl['Date'], format='mixed', errors='coerce')
    gl = gl.dropna(subset=['Date'])
    gl['month']   = gl['Date'].dt.strftime('%b')
    gl['is_home'] = gl.get('Home_Away', pd.Series('', index=gl.index)) == 'Home'
    gl['is_win']  = gl['W/L'].astype(str).str.startswith('W')

    grp = gl.groupby('month', sort=False).agg(
        games=('Date', 'count'),
        home_games=('is_home', 'sum'),
        wins=('is_win', 'sum'),
    ).reset_index()
    grp['away_games'] = grp['games'] - grp['home_games']
    grp['win_pct']    = (grp['wins'] / grp['games']).round(3)

    month_order = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct']
    grp['_order'] = grp['month'].map({m: i for i, m in enumerate(month_order)})
    grp = grp.sort_values('_order').drop(columns=['_order', 'wins'])
    return grp[['month', 'games', 'home_games', 'away_games', 'win_pct']]


def _stats_to_pitcher_adj(custom_stats: dict, raw_dir: str, tex25: dict | None = None) -> dict | None:
    """sv_pct / onerun_wp → pitcher_adjustments multiplier dict 변환.

    sv_pct delta → closer K%/BB% 조정 (세이브 실패 원인: 주자 허용)
    onerun_wp delta → setup + middle_reliever K%/BB% 조정 (접전 불펜 전체)

    calibration:
        +0.10 sv_pct  → closer K_pct × 1.25, BB_pct × 0.85
        +0.10 onerun  → 불펜 K_pct × 1.20, BB_pct × 0.85
    """
    if not custom_stats:
        return None
    if tex25 is None:
        defaults = _get_scenario_defaults(raw_dir)
        tex25 = defaults.get("tex25", {})
    baseline_sv    = float(tex25.get("sv_pct",    0.70))
    baseline_onerun = float(tex25.get("onerun_wp", 0.50))

    delta_sv     = float(custom_stats.get("sv_pct",    baseline_sv))    - baseline_sv
    delta_onerun = float(custom_stats.get("onerun_wp", baseline_onerun)) - baseline_onerun

    adj: dict = {}
    if abs(delta_sv) > 1e-4:
        adj["closer"] = {
            "K_pct":  max(0.50, min(2.0, 1.0 + delta_sv * 2.5)),
            "BB_pct": max(0.50, min(2.0, 1.0 - delta_sv * 1.5)),
        }
    if abs(delta_onerun) > 1e-4:
        mult = {
            "K_pct":  max(0.50, min(2.0, 1.0 + delta_onerun * 2.0)),
            "BB_pct": max(0.50, min(2.0, 1.0 - delta_onerun * 1.5)),
        }
        adj["setup"]            = mult
        adj["middle_reliever"]  = mult
    return adj or None


def _get_integrated_sim_result(raw_dir: str, n_sims: int, custom_boosts: dict,
                                custom_stats: dict | None = None,
                                track_player_stats: bool = True) -> dict:
    """integrated_sim(markov_pitching 포함)으로 실행 — 선수별 기록 포함.

    최대 50시즌으로 제한. 선수별 타자·투수 성적 DataFrame을 player_projection /
    pitcher_projection으로 반환.
    ML 잔차 모델(_predict_residual) 사후 적용 — Markov 순수 시뮬은 Pythagorean 수준에서 수렴하므로
    ML이 포착하는 잔차 요인(sv_pct, onerun_wp 등)을 시나리오 stats 기준으로 보정.
    """
    from integrated_sim import run_integrated_simulation, _state as _isim_state
    from simulator import _predict_residual

    n_seasons = n_sims
    hitter_adj  = {'per_player': custom_boosts} if custom_boosts else None
    pitcher_adj = _stats_to_pitcher_adj(custom_stats, raw_dir) if custom_stats else None

    season_df, monthly_df, batter_df, pitcher_df = run_integrated_simulation(
        n_seasons=n_seasons,
        hitter_adjustments=hitter_adj,
        pitcher_adjustments=pitcher_adj,
        raw_dir=Path(raw_dir),
        track_player_stats=True,
    )

    wins = season_df['W'].values.astype(float)
    rs_vals = season_df['RS'].values.astype(float)
    ra_vals = season_df['RA'].values.astype(float)

    # ML 잔차 모델 사후 적용
    # bundle은 run_integrated_simulation 내 _ensure_loaded()로 이미 로드된 상태
    bundle = _isim_state.get('bundle')
    if bundle is not None:
        effective_stats = dict(bundle.tex25)
        if custom_stats:
            effective_stats.update(custom_stats)
        residual_bonus = _predict_residual(bundle, effective_stats)
        wins = wins + residual_bonus

    actual_rs = float(bundle.tex25["RS"]) if bundle is not None else None
    actual_ra = float(bundle.tex25["RA"]) if bundle is not None else None
    actual_w  = float(bundle.tex25["W"])  if bundle is not None else None

    distribution = pd.DataFrame({"wins": wins})
    return {
        "distribution": distribution,
        "summary": {
            "mean": float(wins.mean()),
            "median": float(np.median(wins)),
            "p10": float(np.quantile(wins, 0.10)),
            "p90": float(np.quantile(wins, 0.90)),
            "over_81_5": float((wins >= 82).mean()),
            "over_87_5": float((wins >= 88).mean()),
            "integrated_n_seasons": n_seasons,
            "rs_mean": float(rs_vals.mean()),
            "ra_mean": float(ra_vals.mean()),
            "actual_rs": actual_rs,
            "actual_ra": actual_ra,
            "actual_w":  actual_w,
        },
        "monthly_summary":   monthly_df,
        "schedule_context":  _build_schedule_context(raw_dir),
        "series_summary":    pd.DataFrame(),
        "player_projection":  batter_df,
        "pitcher_projection": pitcher_df,
    }


_PHASE8_CONFIGS: dict[str, dict] = {
    'phase8_max': {
        'hitter':  {'team': {'single_mult': 1.136, 'k_mult': 0.920}},
        'pitcher': {'starter': {'HR_pct': 0.974}, 'closer': {'K_pct': 1.075}},
    },
    'phase8_recovery': {
        'hitter':  {'team': {'single_mult': 1.181}},
        'pitcher': {'starter': {'HR_pct': 0.960}, 'closer': {'K_pct': 1.079}},
    },
    'phase8_safe': {
        'hitter':  {'team': {'single_mult': 1.136, 'hr_mult': 1.115}},
        'pitcher': {'starter': {'HR_pct': 0.976}},
    },
}


# Pareto 후보별 tex25 대비 stat 증감 (v5 NSGA-II Pareto front에서 추출)
_PARETO_STAT_DELTAS: dict[str, dict] = {
    'pareto_aggressive':   {'sv_pct': +0.107, 'ir_pct': +0.067, 'onerun_wp': +0.131, 'xi_wp': +0.220, 'HR9': -0.265, 'BB9': -0.582},
    'pareto_balanced':     {'sv_pct': +0.032, 'ir_pct': +0.067, 'onerun_wp': +0.115, 'xi_wp': +0.220, 'HR9': -0.271, 'BB9': -0.562},
    'pareto_conservative': {'sv_pct': +0.045, 'ir_pct': +0.007, 'onerun_wp': +0.109, 'xi_wp': +0.176, 'HR9': -0.102, 'BB9': +0.099},
}


@st.cache_data(show_spinner=False)
def get_live_scenario_results(raw_dir: str, n_sims: int = 10) -> dict:
    """Baseline + Phase 8 3종 + Pareto 3종을 동일한 통합 Markov 시뮬 기준으로 재계산.

    모든 delta는 같은 baseline_W를 기준으로 계산되므로 직접 비교 가능.
    """
    from integrated_sim import run_integrated_simulation, _state as _isim_state
    from simulator import _predict_residual

    n_seasons = min(n_sims, 50)

    # tex25를 bundle과 독립적으로 먼저 로드 (Pareto custom_stats 계산용)
    defaults = _get_scenario_defaults(raw_dir)
    tex25: dict = defaults.get("tex25", {})

    def _run_mean(hitter_adj, pitcher_adj) -> float:
        season_df, _ = run_integrated_simulation(
            n_seasons=n_seasons,
            hitter_adjustments=hitter_adj,
            pitcher_adjustments=pitcher_adj,
            raw_dir=Path(raw_dir),
            track_player_stats=False,
        )
        wins = season_df['W'].values.astype(float)
        bundle = _isim_state.get('bundle')
        if bundle is not None:
            wins = wins + _predict_residual(bundle, dict(bundle.tex25))
        return float(wins.mean())

    def _run_mean_with_custom(custom_stats: dict) -> float:
        pitcher_adj = _stats_to_pitcher_adj(custom_stats, raw_dir, tex25=tex25)
        season_df, _ = run_integrated_simulation(
            n_seasons=n_seasons,
            hitter_adjustments=None,
            pitcher_adjustments=pitcher_adj,
            raw_dir=Path(raw_dir),
            track_player_stats=False,
        )
        wins = season_df['W'].values.astype(float)
        bundle = _isim_state.get('bundle')
        if bundle is not None:
            effective = {**dict(bundle.tex25), **custom_stats}
            wins = wins + _predict_residual(bundle, effective)
        return float(wins.mean())

    baseline_W = _run_mean(None, None)
    out: dict = {'baseline_W': round(baseline_W, 1)}

    # Phase 8
    for key, cfg in _PHASE8_CONFIGS.items():
        try:
            pred_W = _run_mean(cfg.get('hitter'), cfg.get('pitcher'))
            out[key] = {
                'predicted_W': round(pred_W, 1),
                'delta':       round(pred_W - baseline_W, 2),
            }
        except Exception:
            pass

    # Pareto — tex25에서 직접 custom_stats 계산 (bundle 의존 없음)
    for key, deltas in _PARETO_STAT_DELTAS.items():
        try:
            custom_stats = {
                feat: round(float(tex25.get(feat, 0.0)) + delta, 4)
                for feat, delta in deltas.items()
            }
            pred_W = _run_mean_with_custom(custom_stats)
            out[key] = {
                'predicted_W': round(pred_W, 1),
                'delta':       round(pred_W - baseline_W, 2),
            }
        except Exception:
            pass

    return out


@st.cache_data(show_spinner=False)
def get_simulation_result(
    raw_dir: str,
    scenario_name: str,
    n_sims: int,
    custom_stats: dict | None = None,
    custom_boosts: dict | None = None,
    fast_mode: bool = True,
) -> dict:
    # 항상 integrated_sim (markov_pitching 투수 시뮬 포함)
    if not fast_mode:
        return _get_integrated_sim_result(raw_dir, n_sims, custom_boosts, custom_stats=custom_stats)
    return run_simulation(
        raw_dir,
        scenario_name,
        n_sims=n_sims,
        custom_stats=custom_stats,
        custom_boosts=custom_boosts,
        fast_mode=fast_mode,
    )


@st.cache_data(show_spinner=False)
def get_simulation_defaults(raw_dir: str) -> dict:
    return _get_scenario_defaults(raw_dir)


@st.cache_data(show_spinner=False)
def get_simulation_batters(raw_dir: str) -> list[str]:
    return _get_batter_options(raw_dir)


@st.cache_data(show_spinner=False)
def get_scenario_snapshots(raw_dir: str) -> dict:
    return build_scenario_snapshots(raw_dir)


# ── Format helpers ────────────────────────────────────────────
def fmt_pct(value, digits: int = 0) -> str:
    try:
        return f"{float(value):.{digits}%}"
    except Exception:
        return "-"


def fmt_num(value, digits: int = 1, sign: bool = False) -> str:
    try:
        return f"{float(value):+.{digits}f}" if sign else f"{float(value):.{digits}f}"
    except Exception:
        return "-"


# ── Report findings ───────────────────────────────────────────
REPORT_FINDINGS = {
    "Leiter": {
        "role": "Starter",
        "summary": "삼진 vs 볼넷에서 통계적으로 강한 모션 분기는 제한적입니다.",
        "recommendation": "폼 교정보다는 피칭 디자인, 구종 선택, 카운트별 운영을 우선 점검합니다.",
    },
    "Webb": {
        "role": "Starter",
        "summary": "HSS와 Trunk/Hip ratio에서 뚜렷한 차이가 확인된 핵심 코칭 대상입니다.",
        "recommendation": "하체-상체 분리와 릴리스 전 회전 타이밍을 안정화하는 루틴을 설계합니다.",
    },
    "Garcia": {
        "role": "Closer",
        "summary": "세이브 vs 블론 세이브 간 모션 차이가 작아 null finding으로 해석하는 편이 적절합니다.",
        "recommendation": "폼 수정 대신 deployment, 매치업, 구질 조합, 외부 보강 판단과 연결합니다.",
    },
    "Armstrong": {
        "role": "Reliever",
        "summary": "일부 지표 변동은 있으나 전면적인 폼 결함으로 단정하기는 어렵습니다.",
        "recommendation": "짧은 등판 특성을 고려해 워밍업과 등판 전 루틴을 표준화합니다.",
    },
    "Jackson": {
        "role": "Reliever",
        "summary": "sidearm보다는 lateral tilt overhand 패턴으로 재해석하는 것이 적절합니다.",
        "recommendation": "부상 위험 단정 대신 platoon split과 specialist deployment 관점으로 운영합니다.",
    },
}


# ── PDF design tokens (app_260501 색상 시스템) ────────────────
_P_BG         = "#FAFBFC"      # --background
_P_FG         = "#1B2435"      # --foreground
_P_PRIMARY    = "#0D1B33"      # --navy
_P_NAVY_DEEP  = "#071225"      # --navy-deep
_P_NAVY_SOFT  = "#243A5E"      # --navy-soft
_P_PRIMARY_FG = "#FFFFFF"      # header text (white on navy)
_P_CARD       = "#FFFFFF"      # --card
_P_SECONDARY  = "#F3F5F8"      # --muted
_P_MUTED_FG   = "#667085"      # --muted-foreground
_P_BORDER     = "#E4E8EF"      # --border
_P_GRID       = "#E9EDF3"      # --grid  (alternating row)
_P_RED        = "#B31922"      # --rangers-red / --negative
_P_RED_SOFT   = "#D04A52"      # --rangers-red-soft
_P_NEUTRAL    = "#B8BDC7"      # --neutral

REPORT_FINDINGS.update({
    "Leiter": {
        "role": "Starter",
        "summary": "삼진과 볼넷 상황의 모션 차이는 제한적입니다. 제구 문제를 특정 동작 하나로 단정하기보다 구종 선택, 카운트별 접근, 타자 대응을 함께 봐야 합니다.",
        "recommendation": "단일 동작 교정보다는 초구 스트라이크 확보, 불리한 카운트에서의 구종 조합, 좌우 타자별 운영 점검을 우선 권장합니다.",
    },
    "Webb": {
        "role": "Starter",
        "summary": "HSS와 몸통/골반 회전 비율에서 뚜렷한 차이가 확인됩니다. 하이 레버리지 상황에서 하체-상체 연결과 릴리스 타이밍이 흔들렸을 가능성이 있습니다.",
        "recommendation": "하체-상체 분리와 릴리스 전 회전 타이밍을 안정화하는 코칭을 우선 검토하고, 부담이 큰 이닝의 연속 등판은 관리가 필요합니다.",
    },
    "Garcia": {
        "role": "Closer",
        "summary": "세이브와 블론 세이브 상황의 모션 차이는 작습니다. 부진 원인을 투구폼만으로 설명하기보다 배치, 매치업, 구위 변화까지 함께 해석해야 합니다.",
        "recommendation": "마무리 고정 여부보다 상대 타순, 좌우 매치업, 연투 여부에 따른 하이 레버리지 배치를 재검토하는 쪽이 적절합니다.",
    },
    "Armstrong": {
        "role": "Reliever",
        "summary": "일부 지표 변화는 보이지만 전반적인 폼 결함으로 단정하기 어렵습니다. 표본 크기가 작아 해석은 보조 근거로 제한하는 편이 안전합니다.",
        "recommendation": "구위 하락, 컨택 품질, 좌우 스플릿을 같이 확인해 특정 타순 또는 특정 상황 전용 기용 여부를 판단해야 합니다.",
    },
    "Jackson": {
        "role": "Reliever",
        "summary": "사이드암이라기보다 측면 기울기가 큰 오버핸드 패턴에 가깝습니다. 단순 폼 문제보다 특정 타자 유형과의 상성이 더 중요할 수 있습니다.",
        "recommendation": "플래툰 스플릿과 구종 궁합을 기준으로 specialist deployment를 검토하고, 무리한 전천후 기용은 피하는 방향이 적절합니다.",
    },
})

# card accent cycling (navy / red / navy-soft only)
_P_C1         = "#0D1B33"      # navy
_P_C2         = "#B31922"      # rangers-red
_P_C3         = "#243A5E"      # navy-soft
_P_C4         = "#0D1B33"
_P_C5         = "#B31922"
_P_M          = 0.055
_P_CW         = 1 - 2 * _P_M

# ── PDF helpers ───────────────────────────────────────────────
def _configure_pdf_font():
    preferred = ["Malgun Gothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "DejaVu Sans"]
    available = {f.name for f in fm.fontManager.ttflist}
    for font in preferred:
        if font in available:
            mpl.rcParams["font.family"] = font
            break
    mpl.rcParams["axes.unicode_minus"] = False


def _rr(ax, x, y, w, h, r, color, edge="none", lw=0, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={max(r, 0.001)}",
        transform=ax.transAxes,
        facecolor=color, edgecolor=edge, linewidth=lw, alpha=alpha,
    ))


def _pdf_new_page():
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor(_P_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_axis_off()
    return fig, ax


def _pdf_header(ax, title: str, subtitle: str = ""):
    # Full-bleed navy band (y: 0.840 → 1.0)
    _rr(ax, 0, 0.840, 1.0, 0.160, 0.001, _P_PRIMARY)
    # Deeper navy top strip
    _rr(ax, 0, 0.968, 1.0, 0.032, 0.001, _P_NAVY_DEEP)
    # Red rule at bottom of header
    ax.plot([0, 1], [0.840, 0.840], transform=ax.transAxes,
            color=_P_RED, lw=2.0, solid_capstyle="butt")
    # header 세로 중앙: 유효 영역 0.840~0.966, 중심 ≈ 0.903
    ax.text(0.5, 0.938, "TEXAS RANGERS  ·  2025 RESIDUAL ANALYSIS",
            fontsize=7.0, color=_P_NEUTRAL, transform=ax.transAxes,
            va="top", ha="center", fontweight="bold")
    ax.text(0.5, 0.920, title,
            fontsize=18, color=_P_PRIMARY_FG, transform=ax.transAxes,
            va="top", ha="center", fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.889, subtitle,
                fontsize=8.5, color=_P_NEUTRAL, transform=ax.transAxes,
                va="top", ha="center")


def _pdf_chips(ax, y: float, chips: list[tuple[str, str]]):
    """Badge-style pill chips matching app color system."""
    x = _P_M
    for text, tone in chips:
        if tone == "red":
            fg, bg, bd = _P_RED, "#FDF2F2", _P_RED_SOFT
        else:
            fg, bg, bd = _P_NAVY_SOFT, "#EEF2F8", _P_NEUTRAL
        cw = min(0.30, max(0.10, 0.028 + len(text) * 0.009))
        _rr(ax, x, y - 0.013, cw, 0.022, 0.011, bg, edge=bd, lw=0.8)
        ax.text(x + cw / 2, y - 0.002, text,
                fontsize=7.0, color=fg, transform=ax.transAxes,
                va="center", ha="center", fontweight="bold")
        x += cw + 0.008


def _pdf_footer(ax, page_no: int):
    # Top separator line
    ax.plot([0, 1], [0.040, 0.040], transform=ax.transAxes,
            color=_P_BORDER, lw=0.5)
    ax.text(_P_M, 0.025, "TEX 2025  ·  Residual Diagnosis & Decision Candidates",
            fontsize=6.5, color=_P_MUTED_FG, transform=ax.transAxes, va="center")
    # Page number — primary pill
    pn_x = 1 - _P_M - 0.036
    _rr(ax, pn_x, 0.007, 0.034, 0.022, 0.011, _P_PRIMARY)
    ax.text(pn_x + 0.017, 0.018, str(page_no),
            fontsize=7, color=_P_PRIMARY_FG, transform=ax.transAxes,
            va="center", ha="center", fontweight="bold")


def _pdf_card(ax, x: float, y_top: float, w: float, h: float,
              title: str, lines: list[str], accent: str = _P_PRIMARY,
              section_no: int = 0):
    y0 = y_top - h
    # White card with border
    _rr(ax, x, y0, w, h, 0.010, "white", edge=_P_BORDER, lw=0.7)
    # Left accent stripe — tailwind border-l-4 style
    _rr(ax, x, y0 + 0.010, 0.006, h - 0.020, 0.003, accent)
    # Title area — muted background
    title_h = 0.043
    _rr(ax, x, y_top - title_h, w, title_h, 0.010, _P_SECONDARY)
    _rr(ax, x, y_top - title_h, w, title_h / 2, 0.001, _P_SECONDARY)
    ax.plot([x, x + w], [y_top - title_h, y_top - title_h],
            transform=ax.transAxes, color=_P_BORDER, lw=0.5)
    # Title text
    ax.text(x + 0.026, y_top - title_h / 2, title,
            fontsize=10.5, color=_P_PRIMARY,
            transform=ax.transAxes, va="center", fontweight="bold")
    # Section badge
    if section_no:
        badge_w = 0.036
        badge_x = x + w - badge_w - 0.012
        _rr(ax, badge_x, y_top - title_h / 2 - 0.011, badge_w, 0.022, 0.011,
            _P_BORDER)
        ax.text(badge_x + badge_w / 2, y_top - title_h / 2,
                f"{section_no:02d}",
                fontsize=7.0, color=_P_MUTED_FG,
                transform=ax.transAxes, va="center", ha="center", fontweight="bold")

    # _estimate_card_height 와 동일한 _cjk_wrap 기반으로 centering 계산
    def _body_h(lines):
        ch = 0.0
        for ln in lines:
            if ln == "":
                ch += 0.005
            elif ": " in ln and not ln.startswith("- "):
                ch += 0.038
            else:
                ch += len(_cjk_wrap(ln[2:] if ln.startswith("- ") else ln)) * 0.022
        return ch

    available    = h - title_h
    content_used = _body_h(lines)
    top_space    = max(0.010, (available - content_used) / 2)

    y = y_top - title_h - top_space
    row_alt = True
    for line in lines:
        bottom = y0 + 0.004
        if line == "":
            y -= 0.005
            continue
        if ": " in line and not line.startswith("- "):
            label, value = line.split(": ", 1)
            rh = 0.034
            if y - rh < bottom:
                return
            if row_alt:
                _rr(ax, x + 0.016, y - rh + 0.002, w - 0.032, rh - 0.003,
                    0.004, _P_GRID)
            ax.text(x + 0.026, y - rh / 2 + 0.002, label,
                    fontsize=8.0, color=_P_MUTED_FG,
                    transform=ax.transAxes, va="center", fontweight="bold")
            ax.text(x + 0.260, y - rh / 2 + 0.002, value,
                    fontsize=9.0, color=_P_FG,
                    transform=ax.transAxes, va="center")
            y -= rh + 0.004
            row_alt = not row_alt
            continue
        is_bullet = line.startswith("- ")
        bullet = "·  " if is_bullet else ""
        content = line[2:] if is_bullet else line
        for chunk in _cjk_wrap(content):
            if y < bottom:
                return
            ax.text(x + 0.026, y, bullet + chunk,
                    fontsize=9.0, color=_P_FG, transform=ax.transAxes, va="top")
            bullet = "   " if is_bullet else ""
            y -= 0.022


def _split_pdf_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    headings = {
        "Purpose", "Analysis Flow", "Pitcher-Level Summary", "Team Conclusion",
        "Player Info", "Motion Finding", "Top Kinematic Metrics", "Recommendation", "Interpretation Note",
        "Team / League Context", "Residual Summary", "Team Strength / Weakness", "Team Metric Rank",
    }
    sections: list[tuple[str, list[str]]] = []
    current_title = "Summary"
    current_lines: list[str] = []
    for line in lines:
        if line in headings:
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = line
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))
    return sections


def _cjk_wrap(text: str) -> list[str]:
    """한글(2) · 라틴(1) 시각 폭 기준으로 줄바꿈 (기준폭 110 visual units)."""
    MAX_VW = 110
    words = text.split(" ")
    lines: list[str] = []
    cur, cur_w = "", 0
    for word in words:
        vw = sum(2 if "가" <= c <= "퟿" or "一" <= c <= "鿿" else 1 for c in word)
        gap = 1 if cur else 0
        if cur and cur_w + gap + vw > MAX_VW:
            lines.append(cur)
            cur, cur_w = word, vw
        else:
            cur = (cur + " " + word).lstrip() if cur else word
            cur_w += gap + vw
    if cur:
        lines.append(cur)
    return lines or [text]


def _estimate_card_height(body: list[str]) -> float:
    h = 0.043  # title_h
    for line in body:
        if line == "":
            h += 0.005
        elif ": " in line and not line.startswith("- "):
            h += 0.038
        else:
            h += len(_cjk_wrap(line[2:] if line.startswith("- ") else line)) * 0.022
    h += 0.020  # 상하 대칭 여백
    return min(0.75, max(0.10, h))


def _add_pdf_page(pdf: PdfPages, title: str, lines: list) -> int:
    _configure_pdf_font()
    sections = _split_pdf_sections(lines)
    page_no = 1
    fig, ax = _pdf_new_page()
    subtitle = "Residual diagnosis · Motion evidence · Decision candidates"
    _pdf_header(ax, title, subtitle)

    y = 0.824
    accents = [_P_C1, _P_C2, _P_C3, _P_C1, _P_C2, _P_C3, _P_C1, _P_C2]
    for idx, (heading, body) in enumerate(sections):
        height = _estimate_card_height(body)
        if y - height < 0.058:
            _pdf_footer(ax, page_no)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            page_no += 1
            fig, ax = _pdf_new_page()
            _pdf_header(ax, title, "continued")
            ax.plot([_P_M, 1 - _P_M], [0.824, 0.824], transform=ax.transAxes,
                    color=_P_BORDER, lw=0.5)
            y = 0.812
        accent = accents[idx % len(accents)]
        _pdf_card(ax, _P_M, y, _P_CW, height, heading, body,
                  accent=accent, section_no=idx + 1)
        y -= height + 0.016

    _pdf_footer(ax, page_no)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    return page_no


_PITCHER_RADAR_STATS = [
    ("K/9", "K/9", True),
    ("BB/9", "BB/9", False),
    ("HR/9", "HR/9", False),
    ("ERA", "ERA", False),
    ("FIP", "FIP", False),
    ("WHIP", "WHIP", False),
]


def _read_raw_csv(filename: str) -> pd.DataFrame:
    path = RAW_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _match_pitcher_row(player: str) -> pd.Series | None:
    pitchers = _read_raw_csv("texas_pitchers_2025.csv")
    if pitchers.empty or "Name" not in pitchers.columns:
        return None
    mask = pitchers["Name"].astype(str).str.contains(player, case=False, regex=False)
    if not mask.any():
        mask = pitchers.get("NameASCII", pitchers["Name"]).astype(str).str.contains(player, case=False, regex=False)
    if not mask.any():
        return None
    return pitchers.loc[mask].iloc[0]


def _rank_text(value: float, series: pd.Series, higher_is_better: bool) -> str:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty or pd.isna(value):
        return "-"
    rank = int((values > value).sum() + 1) if higher_is_better else int((values < value).sum() + 1)
    top_pct = rank / len(values) * 100
    return f"{rank}/{len(values)}위, 상위 {top_pct:.0f}%"


def _player_percentile_lines(player: str) -> list[str]:
    pitchers = _read_raw_csv("texas_pitchers_2025.csv")
    row = _match_pitcher_row(player)
    if pitchers.empty or row is None:
        return ["- 선수별 원천 기록을 찾지 못했습니다."]
    stat_specs = [
        ("ERA", "ERA", False),
        ("FIP", "FIP", False),
        ("WHIP", "WHIP", False),
        ("K/9", "K/9", True),
        ("BB/9", "BB/9", False),
        ("WAR", "WAR", True),
    ]
    lines = []
    for label, col, higher_is_better in stat_specs:
        if col not in pitchers.columns:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        lines.append(f"- {label}: {value:.2f} / 팀 내 {_rank_text(value, pitchers[col], higher_is_better)}")
    lines.append("- 리그 내 선수 단위 백분위는 현재 원천 파일에 전체 MLB 투수 목록이 없어 계산하지 않았습니다.")
    lines.append("- 대신 다음 레이더 차트에서 MLB 팀 평균을 점선 기준으로 두고, 해당 선수의 기록을 실선으로 비교합니다.")
    return lines


def _team_metric_lines() -> list[str]:
    teams = _read_raw_csv("mlb_teams_2025_pitching.csv")
    if teams.empty or "Team" not in teams.columns:
        return ["- 팀 투수 기록 파일을 찾지 못했습니다."]
    tex = teams[teams["Team"].astype(str).str.upper() == "TEX"]
    if tex.empty:
        return ["- TEX 팀 행을 찾지 못했습니다."]
    row = tex.iloc[0]
    stat_specs = [
        ("ERA", "ERA", False),
        ("FIP", "FIP", False),
        ("WHIP", "WHIP", False),
        ("K/9", "K/9", True),
        ("BB/9", "BB/9", False),
        ("HR/9", "HR/9", False),
        ("WAR", "WAR", True),
        ("BS", "BS", False),
    ]
    lines = []
    for label, col, higher_is_better in stat_specs:
        if col not in teams.columns:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        lines.append(f"- {label}: {value:.2f} / MLB 30팀 중 {_rank_text(value, teams[col], higher_is_better)}")
    return lines


def _team_strength_lines() -> list[str]:
    teams = _read_raw_csv("mlb_teams_2025_pitching.csv")
    if teams.empty or "Team" not in teams.columns:
        return ["- 팀 강점/약점 계산에 필요한 데이터가 없습니다."]
    tex = teams[teams["Team"].astype(str).str.upper() == "TEX"]
    if tex.empty:
        return ["- TEX 팀 행을 찾지 못했습니다."]
    row = tex.iloc[0]
    strengths = []
    weaknesses = []
    for label, col, higher_is_better in [
        ("실점 억제(ERA)", "ERA", False),
        ("수비 무관 성과(FIP)", "FIP", False),
        ("주자 출루 억제(WHIP)", "WHIP", False),
        ("피홈런 억제(HR/9)", "HR/9", False),
        ("볼넷 억제(BB/9)", "BB/9", False),
        ("탈삼진(K/9)", "K/9", True),
        ("누적 투수 가치(WAR)", "WAR", True),
        ("블론 세이브(BS)", "BS", False),
    ]:
        if col not in teams.columns:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        values = pd.to_numeric(teams[col], errors="coerce").dropna()
        rank = int((values > value).sum() + 1) if higher_is_better else int((values < value).sum() + 1)
        text = f"{label}: {value:.2f}, MLB {rank}/30위"
        if rank <= 10:
            strengths.append(text)
        elif rank >= 21:
            weaknesses.append(text)
    lines = ["강점"]
    lines.extend([f"- {item}" for item in strengths[:4]] or ["- 상위권으로 뚜렷하게 잡히는 단일 지표는 제한적입니다."])
    lines.append("약점 또는 점검 지점")
    lines.extend([f"- {item}" for item in weaknesses[:4]] or ["- 리그 최하위권 지표는 많지 않지만, 하이 레버리지 운영에서 실제 승수 손실이 커졌습니다."])
    lines.append("특징")
    lines.append("- 평균적인 투수 지표만 보면 81승보다 높은 기대 승수를 설명할 여지가 있으나, 접전·세이브 실패·득실 타이밍에서 실제 승수와의 차이가 커졌습니다.")
    return lines


def _add_player_radar_page(pdf: PdfPages, player: str, page_no: int = 2):
    _configure_pdf_font()
    row = _match_pitcher_row(player)
    teams = _read_raw_csv("mlb_teams_2025_pitching.csv")
    if row is None or teams.empty:
        return
    fig, ax = _pdf_new_page()
    _pdf_header(ax, f"{player} Stat Context", "team percentile and MLB average radar")
    ax.text(_P_M, 0.805, "팀 내 위치와 리그 평균 대비 스탯 비교",
            fontsize=14, color=_P_PRIMARY, transform=ax.transAxes,
            va="top", fontweight="bold")
    ax.text(_P_M, 0.775,
            "레이더 차트는 값이 바깥쪽일수록 좋은 방향으로 정규화했습니다. 점선은 MLB 팀 평균, 실선은 해당 선수입니다.",
            fontsize=8.5, color=_P_MUTED_FG, transform=ax.transAxes, va="top")

    lines = _player_percentile_lines(player)
    _pdf_card(ax, _P_M, 0.720, _P_CW, 0.250, "팀 내 상위 비율", lines[:7], accent=_P_RED, section_no=1)

    labels = [item[0] for item in _PITCHER_RADAR_STATS]
    player_scores = []
    league_scores = []
    for _, col, higher_is_better in _PITCHER_RADAR_STATS:
        if col not in teams.columns or col not in row.index:
            player_scores.append(np.nan)
            league_scores.append(np.nan)
            continue
        values = pd.to_numeric(teams[col], errors="coerce").dropna()
        vmin, vmax = values.min(), values.max()
        denom = vmax - vmin if vmax != vmin else 1
        player_value = float(row[col])
        league_value = float(values.mean())
        if higher_is_better:
            p_score = (player_value - vmin) / denom
            l_score = (league_value - vmin) / denom
        else:
            p_score = (vmax - player_value) / denom
            l_score = (vmax - league_value) / denom
        player_scores.append(float(np.clip(p_score, 0, 1)))
        league_scores.append(float(np.clip(l_score, 0, 1)))

    valid = [i for i, (p, l) in enumerate(zip(player_scores, league_scores)) if not (np.isnan(p) or np.isnan(l))]
    labels = [labels[i] for i in valid]
    player_scores = [player_scores[i] for i in valid]
    league_scores = [league_scores[i] for i in valid]

    radar = fig.add_axes([0.16, 0.062, 0.68, 0.315], polar=True)
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    p_values = player_scores + player_scores[:1]
    l_values = league_scores + league_scores[:1]
    radar.plot(angles, l_values, color=_P_MUTED_FG, linestyle="--", linewidth=2.0, label="MLB 평균")
    radar.plot(angles, p_values, color=_P_RED, linewidth=2.4, label=player)
    radar.fill(angles, p_values, color=_P_RED, alpha=0.12)
    radar.set_xticks(angles[:-1])
    radar.set_xticklabels(labels, fontsize=9, color=_P_FG)
    radar.set_yticks([0.25, 0.5, 0.75, 1.0])
    radar.set_yticklabels(["25", "50", "75", "100"], fontsize=7, color=_P_MUTED_FG)
    radar.set_ylim(0, 1)
    radar.grid(color=_P_BORDER, linewidth=0.8)
    radar.spines["polar"].set_color(_P_BORDER)
    radar.legend(loc="upper right", bbox_to_anchor=(1.20, 1.15), frameon=False, fontsize=8)

    _pdf_footer(ax, page_no)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _legacy_build_player_report_pdf(player: str) -> bytes:
    info = data["meta"]["pitchers"].get(player, {})
    finding = REPORT_FINDINGS.get(player, {})
    pitcher_df = data["pitcher_ag"][data["pitcher_ag"]["player"] == player].copy()
    top_metrics = []
    if not pitcher_df.empty:
        top = pitcher_df.reindex(pitcher_df["cohens_d"].abs().sort_values(ascending=False).index).head(5)
        for _, row in top.iterrows():
            top_metrics.append(
                f"- {row['label']}: diff {row['diff']:.2f}, Cohen's d {row['cohens_d']:.2f}, p {row['u_p']:.3f}"
            )
    lines = [
        "Purpose",
        "2025 Texas Rangers의 실제 81승과 Pythagorean 기대 승수 90.06승 사이의 -9.06승 잔차를 설명하기 위한 선수별 보조 보고서입니다.",
        "이 문서는 Simulation 의사결정 후보를 해석할 때 코칭 가능 영역과 운영·보강 영역을 구분하는 근거로 사용합니다.",
        "",
        "Player Info",
        f"Player: {player}",
        f"Role: {finding.get('role', info.get('role', '-'))}",
        f"Case: {info.get('situation', '-')}",
        f"Sample: n={info.get('n_a', '-')}/{info.get('n_b', '-')}",
        "",
        "Motion Finding",
        finding.get("summary", "-"),
        "",
        "Top Kinematic Metrics",
        *(top_metrics or ["- 통계 데이터가 없습니다."]),
        "",
        "Recommendation",
        finding.get("recommendation", "-"),
        "",
        "Interpretation Note",
        "이 보고서는 모션 분석을 잔차 분석의 하위 근거로 사용합니다. 단일 선수의 폼만으로 -9.06승 전체를 설명하지 않고, 경기력·선수 운영·Simulation 의사결정 후보와 함께 해석해야 합니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, f"TEX 2025 Player Report - {player}", lines)
    return buffer.getvalue()


def _legacy_build_team_report_pdf() -> bytes:
    lines = [
        "Purpose",
        "2025 Texas Rangers는 실제 81승, Pythagorean 기대 승수 90.06승으로 -9.06승 잔차를 기록했습니다.",
        "본 대시보드는 이 차이를 설명하기 위해 경기력 분석, 선수 분석, 하이 레버리지 부진 대표 케이스 모션 분석, Simulation 의사결정 후보 비교를 순서대로 연결합니다.",
        "",
        "Analysis Flow",
        "1. 경기력 분석: 득실과 실제 승패가 어긋난 구간, 1점차/연장/세이브 상황 등 잔차가 커진 경기 맥락을 확인합니다.",
        "2. 선수 분석: 부상, 타격/투수 지표, 선수별 projection 변화가 팀 승수에 미친 조건을 분리합니다.",
        "3. 하이 레버리지 부진 대표 케이스 모션 분석: 코칭 가능 영역과 운영·보강 영역을 3D 키네마틱으로 구분합니다.",
        "4. Simulation: 수동 시나리오와 Grid/Pareto 후보를 같은 기준으로 비교합니다.",
        "5. 의사결정 후보 순위표: 승수 개선 폭, 예측 안정성, 실행 관점의 해석을 함께 봅니다.",
        "6. AI Agent: 시나리오 조회, 최적화 요약, 팀 비교 질의를 보조합니다.",
        "",
        "Pitcher-Level Summary",
    ]
    for player, finding in REPORT_FINDINGS.items():
        lines.append(f"- {player}: {finding['summary']} Recommendation: {finding['recommendation']}")
    lines.extend([
        "",
        "Team Conclusion",
        "하이 레버리지 부진 대표 케이스 모션 분석은 잔차 -9.06승의 전체 원인이 아니라, 선수/운영 분석을 더 구체화하는 하위 레이어입니다.",
        "Webb처럼 코칭 가능한 폼 분기가 있는 선수와 Garcia처럼 모션 외 요인을 우선 검토해야 하는 선수를 분리하는 것이 핵심입니다.",
        "최종 판단은 Simulation의 의사결정 후보 순위표를 중심으로, 승수 개선 폭과 예측 안정성, 선수별 projection, 하이 레버리지 부진 케이스 모션 근거를 함께 종합해야 합니다.",
    ])
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, "TEX 2025 Team Summary Report", lines)
    return buffer.getvalue()


# ── Data loading ──────────────────────────────────────────────
# Report builders are redefined here to keep the dashboard copy readable and
# to add the percentile/radar pages used by the final presentation.
def build_player_report_pdf(player: str) -> bytes:
    info = data["meta"]["pitchers"].get(player, {})
    finding = REPORT_FINDINGS.get(player, {})
    pitcher_df = data["pitcher_ag"][data["pitcher_ag"]["player"] == player].copy()
    raw_row = _match_pitcher_row(player)
    display_name = str(raw_row["Name"]) if raw_row is not None and "Name" in raw_row.index else player
    top_metrics = []
    if not pitcher_df.empty:
        top = pitcher_df.reindex(pitcher_df["cohens_d"].abs().sort_values(ascending=False).index).head(5)
        for _, row in top.iterrows():
            top_metrics.append(
                f"- {row['label']}: diff {row['diff']:.2f}, Cohen's d {row['cohens_d']:.2f}, p {row['u_p']:.3f}"
            )
    lines = [
        "Purpose",
        "2025 Texas Rangers는 실제 81승, Pythagorean 기대 승수 90.06승으로 -9.06승의 잔차를 기록했습니다.",
        "이 문서는 하이 레버리지 상황에서 부진했던 투수 중 대표 케이스를 선수 단위로 정리하고, 팀 내 위치와 리그 평균 대비 특징을 함께 보여줍니다.",
        "",
        "Player Info",
        f"Player: {display_name}",
        f"Role: {finding.get('role', info.get('role', '-'))}",
        f"Case: {info.get('situation', '-')}",
        f"Sample: n={info.get('n_a', '-')}/{info.get('n_b', '-')}",
        "",
        "Team / League Context",
        *_player_percentile_lines(player),
        "",
        "Motion Finding",
        finding.get("summary", "-"),
        "",
        "Top Kinematic Metrics",
        *(top_metrics or ["- 통계적으로 비교 가능한 키네마틱 지표가 없습니다."]),
        "",
        "Recommendation",
        finding.get("recommendation", "-"),
        "",
        "Interpretation Note",
        "선수별 리포트는 잔차 분석의 하위 근거입니다. 한 선수의 모션만으로 -9.06승 전체를 설명하지 않고, 경기 운영·선수 상태·Simulation 의사결정 후보와 함께 해석해야 합니다.",
        "- 종합 결론: 이 선수는 하이 레버리지 부진 대표 케이스 중 하나이며, 코칭 가능한 동작 문제와 운영·배치로 풀어야 할 문제를 분리해서 판단하는 것이 핵심입니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        last_page = _add_pdf_page(pdf, f"TEX 2025 Player Report - {player}", lines)
        _add_player_radar_page(pdf, player, page_no=last_page + 1)
    return buffer.getvalue()


def build_team_report_pdf() -> bytes:
    player_lines = [
        f"- {player}: {finding['summary']} Recommendation: {finding['recommendation']}"
        for player, finding in REPORT_FINDINGS.items()
    ]
    lines = [
        "Purpose",
        "2025 Texas Rangers는 실제 81승, Pythagorean 기대 승수 90.06승으로 -9.06승의 잔차를 기록했습니다.",
        "팀 전체 리포트는 잔차 요약, 팀 투수 지표의 장단점, 하이 레버리지 부진 대표 케이스, 최종 운영 결론을 한 흐름으로 정리합니다.",
        "",
        "Residual Summary",
        "Actual Wins: 81.0",
        "Pythagorean Expected Wins: 90.06",
        "Residual: -9.06 wins",
        "- 득실점 기반 기대 승수보다 실제 승수가 약 9승 낮았습니다.",
        "- 이 차이는 전력 자체가 약했다는 뜻만은 아니며, 접전 운영·세이브 실패·승패 타이밍 같은 경기 맥락을 함께 봐야 합니다.",
        "",
        "Team Strength / Weakness",
        *_team_strength_lines(),
        "",
        "Team Metric Rank",
        *_team_metric_lines(),
        "",
        "Pitcher-Level Summary",
        *player_lines,
        "",
        "Team Conclusion",
        "하이 레버리지 부진 대표 케이스의 모션 분석은 잔차 -9.06승의 전체 원인이 아니라, 운영 판단을 더 구체화하는 하위 근거입니다.",
        "Webb처럼 코칭 가능한 신호가 보이는 선수와 Garcia처럼 모션보다 배치·매치업 점검이 중요한 선수를 분리해야 합니다.",
        "최종 판단은 Simulation의 의사결정 후보 순위표를 중심으로, 승수 개선 폭과 예측 안정성, 선수별 projection, 하이 레버리지 부진 케이스의 모션 근거를 함께 종합해야 합니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, "TEX 2025 Team Summary Report", lines)
    return buffer.getvalue()


@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    return {
        'pitcher_ag': pd.read_csv(base / "pitcher_stats_ag.csv", encoding="utf-8-sig"),
        'pitcher_mb': pd.read_csv(base / "pitcher_stats_mb.csv", encoding="utf-8-sig"),
        'model_comp': pd.read_csv(base / "model_comparison.csv", encoding="utf-8-sig"),
        'model_sum':  pd.read_csv(base / "model_summary.csv",    encoding="utf-8-sig"),
        'meta': json.load(open(base / "meta.json", encoding="utf-8")),
    }


data = load_data()
