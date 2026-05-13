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
BASE_DIR = Path(__file__).resolve().parent  # Final/
PROJECT_ROOT = BASE_DIR.parent             # repo root/

DATA_DIR = BASE_DIR / "data"
RAW_DIR = BASE_DIR / "data_raw"
ASSETS = BASE_DIR / "assets"

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
        <strong>{title}</strong>
        <div class="finding-box-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


def glass_note(body):
    st.markdown(f'<div class="glass-card">{body}</div>', unsafe_allow_html=True)


def glossary_box(title: str, terms: dict[str, str], mb: int | None = None, show_header: bool = True):
    if "?" in title or "吏" in title:
        title = "키네마틱 지표 용어"
    rows = "".join(
        f'<div style="display:grid;grid-template-columns:minmax(88px,0.28fr) 1fr;gap:10px;'
        f'padding:8px 0;border-top:1px solid rgba(13,27,51,0.08);">'
        f'<div style="font-weight:800;color:#0D1B33;">{term}</div>'
        f'<div style="color:#475569;">{desc}</div></div>'
        for term, desc in terms.items()
    )
    mb_style = f"margin-bottom:{mb}px;" if mb is not None else ""
    header_html = f"""
        <div class="chart-title">{title}</div>
        <div class="chart-caption" style="margin-bottom:18px; padding-bottom:2px;">
            표와 그래프를 읽기 전에 필요한 용어만 짧게 정리했습니다.
        </div>""" if show_header else ""
    st.markdown(f"""
    <div class="glass-card" style="{mb_style}">{header_html}
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
    "ERA": "평균 자책점입니다. 투수가 허용한 자책점을 9이닝 기준으로 환산한 값입니다.",
    "FIP": "수비 영향을 줄이고 삼진, 볼넷, 피홈런 중심으로 투수 성과를 본 지표입니다.",
    "WHIP": "이닝당 허용한 안타와 볼넷 수입니다. 낮을수록 주자를 덜 내보냅니다.",
    "WAR": "대체 선수 대비 승리 기여도입니다. 높을수록 팀 승리에 더 많이 기여했다는 뜻입니다.",
    "Clutch": "득점 중요도가 높은 장면에서 얼마나 잘 막아냈는지 보는 지표입니다. 양수면 결정적 순간에 강했다는 뜻, 음수면 중요한 장면에서 오히려 더 많이 실점했다는 뜻입니다.",
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
    머신러닝 잔차 모델(_predict_residual) 사후 적용 — Markov 순수 시뮬은 피타고리안 수준에서 수렴하므로
    머신러닝이 포착하는 잔차 요인(sv_pct, onerun_wp 등)을 시나리오 stats 기준으로 보정.
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

    # 머신러닝 잔차 모델 사후 적용
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

# ── Kinematic metric descriptions ────────────────────────────
REPORT_FINDINGS = {
    player: REPORT_FINDINGS[player]
    for player in ["Webb", "Leiter", "Garcia", "Armstrong", "Jackson"]
}

_METRIC_INFO: dict[str, tuple[str, str]] = {
    "hip_peak_dps":    ("골반 피크 각속도 XZ", "골반이 XZ 평면에서 최대 회전 속도에 도달하는 정도. 낮을수록 하체 구동력이 약함."),
    "trunk_peak_dps":  ("몸통 피크 각속도 XZ", "상체(몸통) 회전의 최대 속도. 투구 파워와 직결되는 지표."),
    "hip_3d_dps":      ("골반 피크 각속도 3D", "3차원 공간 전체를 고려한 골반 회전 속도."),
    "trunk_3d_dps":    ("몸통 피크 각속도 3D", "3차원 공간 전체를 고려한 몸통 회전 속도."),
    "trunk_hip_ratio": ("몸통/골반 각속도 비",  "하체 대비 상체 회전 속도 비율. 클수록 에너지 전달 효율이 높음."),
    "timing_diff_ms":  ("골반-어깨 타이밍 차",  "골반과 어깨 최대 회전 간 시간 차(ms). 양수면 골반이 먼저 회전."),
    "hss_at_fp_deg":   ("HSS @ FP",             "Foot Plant 시점의 골반-어깨 분리각(°). 투구 메커니즘의 핵심 지표."),
    "hss_max_deg":     ("HSS 최대",              "투구 전체 동작에서 측정된 최대 골반-어깨 분리각(°)."),
}


def _cohens_d_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:
        return "미미한 차이"
    elif ad < 0.5:
        return "작은 효과크기"
    elif ad < 0.8:
        return "중간 효과크기"
    else:
        return "큰 효과크기 (코칭 가능 신호)"


def _sig_label(p: float) -> str:
    if p < 0.05:
        return "통계적 유의"
    elif p < 0.10:
        return "경계 유의"
    else:
        return "비유의"


_P_M          = 0.055
_P_CW         = 1 - 2 * _P_M

# ── 섹션별 시맨틱 accent 색상 ─────────────────────────────────
_PDF_SECTION_ACCENT: dict[str, str] = {
    # 개요·요약
    "Purpose":                       "#243A5E",
    "Residual Summary":              "#B31922",
    # 타격
    "Batting Stats":                 "#1D6FA4",
    # 수비
    "Defense & Contact Suppression": "#1A6B5A",
    # 투수
    "Team Strength / Weakness":      "#0D1B33",
    "Pitching Staff Overview":       "#243A5E",
    "Pitching Metric Rank":          "#0D1B33",
    # 월별·이탈
    "Monthly Record":                "#5A3E8C",
    "Key Player Absences":           "#B31922",
    # 불펜·마무리
    "Closer Role Transition":        "#B31922",
    "Inning ERA by Role":            "#0D1B33",
    "Bullpen Save Situation":        "#B31922",
    "Clutch Performance":            "#243A5E",
    # 결론
    "Team Conclusion":               "#0D1B33",
    # 선수 보고서
    "Season Stats":                  "#243A5E",
    "Team / League Context":         "#0D1B33",
    "하이 레버리지 & Clutch":        "#B31922",
    "Motion Finding":                "#0D1B33",
    "Kinematic Analysis Detail":     "#243A5E",
    "OpenBiomechanics Reference":    "#0D9488",
    "Action Priority":               "#B31922",
    "Recommendation":                "#0D1B33",
    "Interpretation Note":           "#667085",
}

# ── PDF helpers ───────────────────────────────────────────────
def _configure_pdf_font():
    # 우선순위: macOS → Windows → Linux(Streamlit Cloud) → fallback
    preferred = [
        "AppleGothic",          # macOS 기본
        "Apple SD Gothic Neo",  # macOS 대안
        "Malgun Gothic",        # Windows
        "NanumGothic",          # Linux (fonts-nanum 설치 후 — 공백 없는 이름)
        "Nanum Gothic",         # Linux (다른 표기)
        "Noto Sans CJK KR",     # Linux (fonts-noto-cjk 설치 후)
        "Noto Sans KR",
        "DejaVu Sans",          # 최후 (한국어 깨짐)
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = None
    for font in preferred:
        if font in available:
            mpl.rcParams["font.family"] = font
            chosen = font
            break
    # 폰트 매칭 실패 시 fc-list로 시스템에서 찾아 등록 시도 (Streamlit Cloud 안전망)
    if chosen is None or chosen == "DejaVu Sans":
        import subprocess, os
        try:
            result = subprocess.run(["fc-list", ":lang=ko", "file"],
                                     capture_output=True, text=True, timeout=5)
            for path in (line.split(":")[0].strip() for line in result.stdout.splitlines() if line):
                if path and os.path.exists(path):
                    fm.fontManager.addfont(path)
                    chosen = fm.FontProperties(fname=path).get_name()
                    mpl.rcParams["font.family"] = chosen
                    break
        except Exception:
            pass
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
    # 네이비 풀블리드 헤더 밴드
    _rr(ax, 0, 0.840, 1.0, 0.160, 0.001, _P_PRIMARY)
    # 상단 짙은 스트립
    _rr(ax, 0, 0.968, 1.0, 0.032, 0.001, _P_NAVY_DEEP)
    # 하단 레드 라인 (굵기 강화)
    ax.plot([0, 1], [0.840, 0.840], transform=ax.transAxes,
            color=_P_RED, lw=3.0, solid_capstyle="butt")
    # 좌측 레드 accent 블록
    _rr(ax, 0, 0.840, 0.006, 0.128, 0.001, _P_RED)
    # 로고/브랜드 kicker
    ax.text(0.5, 0.957, "TEXAS RANGERS  ·  2025 RESIDUAL ANALYSIS",
            fontsize=6.5, color=_P_NEUTRAL, transform=ax.transAxes,
            va="top", ha="center", fontweight="bold")
    # 타이틀
    ax.text(0.5, 0.930, title,
            fontsize=17, color=_P_PRIMARY_FG, transform=ax.transAxes,
            va="top", ha="center", fontweight="bold")
    # 서브타이틀
    if subtitle:
        ax.text(0.5, 0.884, subtitle,
                fontsize=8.0, color=_P_NEUTRAL, transform=ax.transAxes,
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
    # 카드 흰 배경 + 테두리
    _rr(ax, x, y0, w, h, 0.010, "white", edge=_P_BORDER, lw=0.6)
    # 왼쪽 accent 스트라이프
    _rr(ax, x, y0 + 0.010, 0.007, h - 0.020, 0.003, accent)
    # 제목 영역 — accent 미세 틴트 배경
    title_h = 0.044
    import matplotlib.colors as _mc
    try:
        r, g, b = _mc.to_rgb(accent)
        title_bg = (r * 0.06 + 0.94, g * 0.06 + 0.94, b * 0.06 + 0.94)
    except Exception:
        title_bg = _P_SECONDARY
    _rr(ax, x, y_top - title_h, w, title_h, 0.010, title_bg)
    _rr(ax, x, y_top - title_h, w, title_h / 2, 0.001, title_bg)
    ax.plot([x, x + w], [y_top - title_h, y_top - title_h],
            transform=ax.transAxes, color=_P_BORDER, lw=0.4)
    # 제목 텍스트
    ax.text(x + 0.028, y_top - title_h / 2, title,
            fontsize=10.5, color=accent,
            transform=ax.transAxes, va="center", fontweight="bold")
    # 섹션 번호 배지
    if section_no:
        badge_w = 0.036
        badge_x = x + w - badge_w - 0.014
        _rr(ax, badge_x, y_top - title_h / 2 - 0.011, badge_w, 0.022, 0.011, accent)
        ax.text(badge_x + badge_w / 2, y_top - title_h / 2,
                f"{section_no:02d}",
                fontsize=7.0, color="white",
                transform=ax.transAxes, va="center", ha="center", fontweight="bold")

    # _estimate_card_height 와 동일한 _cjk_wrap 기반으로 centering 계산
    def _body_h(lines):
        ch = 0.0
        for ln in lines:
            if ln == "":
                ch += 0.005
            elif ": " in ln and not ln.startswith("- "):
                _, value = ln.split(": ", 1)
                ch += max(0.038, len(_cjk_wrap(value, max_vw=70)) * 0.021 + 0.012)
            else:
                ch += len(_cjk_wrap(ln[2:] if ln.startswith("- ") else ln)) * 0.022
        return ch

    # 본문은 제목 바로 아래에서 시작한다. 세로 가운데 정렬을 쓰면 짧은 섹션에서
    # Kinematic Analysis Detail / OpenBiomechanics Reference처럼 윗부분이 비어 보인다.
    top_space = 0.014

    y = y_top - title_h - top_space
    bottom = y0 + 0.006
    row_alt = True

    def _draw_text(text, tx, ty, size, color, bold=False, italic=False):
        kw = dict(fontsize=size, color=color, transform=ax.transAxes,
                  va="top", fontweight="bold" if bold else "normal")
        if italic:
            kw["fontstyle"] = "italic"
        ax.text(tx, ty, text, **kw)

    for line in lines:
        if y < bottom:
            return
        if line == "":
            y -= 0.006
            continue

        # ── 내부 소제목 ("선발 로테이션 성적:" 등)
        is_subsection = (
            line.endswith(":")
            and not line.startswith("- ")
            and not line.startswith("  ")
            and "→" not in line
            and ": " not in line[:-1]       # "Key: Value" 형식 제외
        )
        if is_subsection:
            if y - 0.032 < bottom:
                return
            # 얇은 구분선
            ax.plot([x + 0.016, x + w - 0.016], [y - 0.003, y - 0.003],
                    transform=ax.transAxes, color=_P_BORDER, lw=0.5)
            _draw_text(line.rstrip(":"), x + 0.026, y - 0.006,
                       8.5, accent, bold=True)
            y -= 0.028
            continue

        # ── Key: Value 표 행
        if ": " in line and not line.startswith("- ") and not line.startswith("  ") and "→" not in line:
            label, value = line.split(": ", 1)
            value_chunks = _cjk_wrap(value, max_vw=70)
            rh = max(0.034, len(value_chunks) * 0.021 + 0.012)
            if y - rh < bottom:
                return
            if row_alt:
                _rr(ax, x + 0.016, y - rh + 0.002, w - 0.032, rh - 0.003,
                    0.004, _P_GRID)
            ax.text(x + 0.026, y - 0.012, label,
                    fontsize=8.0, color=_P_MUTED_FG,
                    transform=ax.transAxes, va="top", fontweight="bold")
            vy = y - 0.010
            for chunk in value_chunks:
                ax.text(x + 0.260, vy, chunk,
                        fontsize=9.0, color=_P_FG,
                        transform=ax.transAxes, va="top")
                vy -= 0.021
            y -= rh + 0.004
            row_alt = not row_alt
            continue

        # ── "→" 해석 포함 라인 분리 렌더링
        if "→" in line:
            arrow_idx = line.index("→")
            main_part  = line[:arrow_idx].rstrip()
            interp     = line[arrow_idx + 1:].strip()

            is_bullet = main_part.startswith("- ")
            indent_x  = x + 0.038 if main_part.startswith("  ") else x + 0.026
            bullet     = "·  " if is_bullet else ""
            content    = main_part[2:].rstrip() if is_bullet else main_part.lstrip()

            # 본문 부분
            for chunk in _cjk_wrap(content):
                if y < bottom:
                    return
                _draw_text(bullet + chunk, indent_x, y, 9.0, _P_FG)
                bullet = "   " if is_bullet else ""
                y -= 0.022

            # 해석 부분 — accent 색, 약간 들여쓰기, 작은 폰트
            if interp:
                for chunk in _cjk_wrap("→ " + interp):
                    if y < bottom:
                        return
                    _draw_text(chunk, indent_x + 0.012, y, 8.2, accent)
                    y -= 0.019
            continue

        # ── 일반 라인 (들여쓰기 2칸 포함)
        is_indented = line.startswith("  ")
        is_bullet   = line.startswith("- ") or line.startswith("  - ")
        if is_indented:
            raw     = line.strip().lstrip("- ").strip()
            indent_x = x + 0.038
            bullet   = "·  " if is_bullet else ""
        else:
            raw     = line[2:] if is_bullet else line
            indent_x = x + 0.026
            bullet   = "·  " if is_bullet else ""

        for chunk in _cjk_wrap(raw):
            if y < bottom:
                return
            _draw_text(bullet + chunk, indent_x, y, 9.0, _P_FG)
            bullet = "   " if is_bullet else ""
            y -= 0.022


def _split_pdf_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    headings = {
        # 공통
        "Purpose", "Team Conclusion", "Recommendation", "Interpretation Note",
        "Player Info",
        # 팀 보고서
        "Residual Summary",
        "Batting Stats",
        "Defense & Contact Suppression",
        "Team Strength / Weakness",
        "Pitching Staff Overview",
        "Pitching Metric Rank",
        "Monthly Record",
        "Key Player Absences",
        "Closer Role Transition",
        "Inning ERA by Role",
        "Bullpen Save Situation",
        "Clutch Performance",
        # 선수 보고서
        "Season Stats",
        "Team / League Context",
        "하이 레버리지 & Clutch",
        "Motion Finding",
        "Kinematic Analysis Detail",
        "OpenBiomechanics Reference",
        "Action Priority",
        # 레거시
        "Analysis Flow", "Pitcher-Level Summary", "Team Metric Rank",
        "Top Kinematic Metrics",
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


def _cjk_wrap(text: str, max_vw: int = 96) -> list[str]:
    """한글(2) · 라틴(1) 시각 폭 기준으로 줄바꿈 (기준폭 110 visual units)."""
    words = text.split(" ")
    lines: list[str] = []
    cur, cur_w = "", 0
    for word in words:
        vw = sum(2 if "가" <= c <= "퟿" or "一" <= c <= "鿿" else 1 for c in word)
        gap = 1 if cur else 0
        if cur and cur_w + gap + vw > max_vw:
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
            h += 0.006
        elif (line.endswith(":") and not line.startswith("- ")
              and not line.startswith("  ") and "→" not in line
              and ": " not in line[:-1]):
            h += 0.028                          # 소제목
        elif ": " in line and not line.startswith("- ") and "→" not in line:
            _, value = line.split(": ", 1)
            h += max(0.038, len(_cjk_wrap(value, max_vw=70)) * 0.021 + 0.012)
        elif "→" in line:
            arrow_idx = line.index("→")
            main = line[:arrow_idx].rstrip()
            interp = line[arrow_idx + 1:].strip()
            content = main[2:].rstrip() if main.startswith("- ") else main.lstrip()
            h += len(_cjk_wrap(content)) * 0.022
            if interp:
                h += len(_cjk_wrap("→ " + interp)) * 0.019
        else:
            raw = line.strip().lstrip("- ").strip()
            h += len(_cjk_wrap(raw)) * 0.022
    h += 0.022  # 상하 여백
    return max(0.10, h)


def _chunk_pdf_body(body: list[str], max_height: float = 0.735) -> list[list[str]]:
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in body:
        trial = current + [line]
        if current and _estimate_card_height(trial) > max_height:
            chunks.append(current)
            current = [line]
        else:
            current = trial
    if current:
        chunks.append(current)
    return chunks or [[]]


def _add_pdf_page(pdf: PdfPages, title: str, lines: list) -> int:
    _configure_pdf_font()
    sections = _split_pdf_sections(lines)
    render_sections: list[tuple[str, list[str]]] = []
    for heading, body in sections:
        chunks = _chunk_pdf_body(body)
        for chunk_idx, chunk in enumerate(chunks):
            chunk_heading = heading if chunk_idx == 0 else f"{heading} (continued)"
            render_sections.append((chunk_heading, chunk))
    page_no = 1
    fig, ax = _pdf_new_page()
    subtitle = "Residual diagnosis · Motion evidence · Decision candidates"
    _pdf_header(ax, title, subtitle)

    _FALLBACK_ACCENTS = [_P_PRIMARY, _P_RED, _P_NAVY_SOFT,
                         _P_PRIMARY, _P_NAVY_SOFT, _P_RED]
    y = 0.824
    for idx, (heading, body) in enumerate(render_sections):
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
        base_heading = heading.replace(" (continued)", "")
        accent = _PDF_SECTION_ACCENT.get(
            base_heading, _FALLBACK_ACCENTS[idx % len(_FALLBACK_ACCENTS)]
        )
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
    n = len(values)
    top_pct = rank / n * 100
    pct_label = f"하위 {100 - top_pct:.0f}%" if top_pct > 50 else f"상위 {top_pct:.0f}%"
    return f"{rank}/{n}위, {pct_label}"


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
    _INTERP: dict[str, dict] = {
        "ERA":  {True:  "실점 억제 리그 최고 수준 → 투수진 기여 우수",
                 False: "실점 허용이 많아 투수진 전반이 팀 승수를 깎는 요인"},
        "FIP":  {True:  "수비 무관 구위 우수 → ERA보다 실질 투구력이 높을 가능성",
                 False: "수비 도움 없이는 ERA보다 실점이 늘어날 구위 수준"},
        "WHIP": {True:  "출루 허용 억제 우수 → 이닝 관리 효율 높음",
                 False: "주자를 자주 허용해 실점 위험 누적"},
        "K/9":  {True:  "탈삼진 능력 우수 → 타구 의존도 낮고 구위로 해결",
                 False: "탈삼진이 적어 타구 처리에 수비 의존도 높음"},
        "BB/9": {True:  "볼넷 허용이 적어 자책점 대비 실점 손실 최소화",
                 False: "볼넷 허용이 많아 무사 진루·빅이닝 위험 높음"},
        "HR/9": {True:  "홈런 허용이 적어 장타 실점 억제 효과적",
                 False: "홈런 허용이 많아 단숨에 점수를 내주는 빈도 높음"},
        "WAR":  {True:  "투수진 누적 가치 우수 → 팀 성적 기여도 높음",
                 False: "투수진 누적 가치 낮음 → 대체 선수 대비 실질 기여 부족"},
        "BS":   {True:  "세이브 실패가 적어 마무리 운영 안정적",
                 False: "세이브 실패 누적 → 잡은 리드를 날리는 빈도가 잔차 확대의 직접 원인"},
    }
    lines = []
    for label, col, higher_is_better in [
        ("ERA", "ERA", False), ("FIP", "FIP", False), ("WHIP", "WHIP", False),
        ("K/9", "K/9", True), ("BB/9", "BB/9", False), ("HR/9", "HR/9", False),
        ("WAR", "WAR", True), ("BS", "BS", False),
    ]:
        if col not in teams.columns:
            continue
        value = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
        rtext = _rank_text(value, teams[col], higher_is_better)
        # Good if rank in top half, bad if bottom half
        all_v = pd.to_numeric(teams[col], errors="coerce").dropna()
        n = len(all_v)
        rank = int((all_v > value).sum() + 1) if higher_is_better else int((all_v < value).sum() + 1)
        is_good = rank <= n // 2
        interp = _INTERP.get(col, {}).get(is_good, "")
        interp_str = f" → {interp}" if interp else ""
        lines.append(f"- {label}: {value:.2f} ({rtext}){interp_str}")
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
    lines.extend([f"- {item}" for item in weaknesses[:4]] or ["- 리그 최하위권 지표는 많지 않지만, 하이 레버리지 투수 운영에서 실제 승수 손실이 커졌습니다."])
    lines.append(
        "- 투수 지표(ERA) 기준으로는 강팀이지만, 블론 세이브 누적과 접전 승률 하락이 실제 승수를"
        " 기대치보다 약 9승 끌어내렸습니다."
    )
    return lines


def _player_season_stats_lines(player: str) -> list[str]:
    row = _match_pitcher_row(player)
    if row is None:
        return ["- 시즌 성적 데이터를 찾지 못했습니다."]

    def _s(col, fmt=".2f"):
        try:
            return f"{float(row[col]):{fmt}}"
        except Exception:
            return "-"

    w = _s("W", "g"); l = _s("L", "g")
    g = _s("G", "g"); gs = _s("GS", "g")
    ip = _s("IP", ".1f")
    sv = _s("SV", "g"); bs = _s("BS", "g")
    lines = [
        f"- 성적: {w}승 {l}패 / {g}게임 ({gs}선발) / {ip}이닝 / {sv}세이브 / {bs}블론",
        f"- ERA {_s('ERA')}  ·  FIP {_s('FIP')}  ·  xERA {_s('xERA')}",
        f"- WHIP {_s('WHIP')}  ·  K/9 {_s('K/9')}  ·  BB/9 {_s('BB/9')}  ·  HR/9 {_s('HR/9')}",
        f"- WAR {_s('WAR')}  ·  BABIP {_s('BABIP')}  ·  LOB% {_s('LOB%')}  ·  GB% {_s('GB%')}",
    ]
    try:
        era_v = float(row["ERA"]); fip_v = float(row["FIP"])
        gap = fip_v - era_v
        if abs(gap) > 0.40:
            msg = ("FIP가 ERA보다 높아 수비·운 도움을 받은 편" if gap > 0
                   else "FIP가 ERA보다 낮아 실제 구위 대비 실점이 많은 편")
            lines.append(f"- ERA-FIP 괴리 {abs(gap):.2f}: {msg}.")
    except Exception:
        pass
    return lines


def _player_situation_lines(player: str) -> list[str]:
    """*하이 레버리지(결정적 순간)* — 1점차·연장·득점권 등 경기 승부에 직접 영향 큰 상황."""
    clutch_df = _read_raw_csv("tex_clutch_pit.csv")
    save_df   = _read_raw_csv("tex_2025_save_situation_splits.csv")
    # 헤더 — 용어 anchor 1줄 (비전공자 친화)
    lines: list[str] = [
        "※ '하이 레버리지'(결정적 순간) = 1점차·연장·득점권 등 *승패에 직결되는 상황*. 그 안에서 이 선수가 얼마나 잘했는지.",
    ]

    clutch_row = None
    if not clutch_df.empty:
        for col in ("NameASCII", "Name"):
            if col in clutch_df.columns:
                mask = clutch_df[col].astype(str).str.contains(player, case=False, regex=False)
                if mask.any():
                    clutch_row = clutch_df.loc[mask].iloc[0]
                    break

    save_row = None
    if not save_df.empty and "Name" in save_df.columns:
        mask = save_df["Name"].astype(str).str.contains(player, case=False, regex=False)
        if mask.any():
            save_row = save_df.loc[mask].iloc[0]

    if clutch_row is not None:
        try:
            pli    = float(clutch_row.get("pLI", "nan"))
            clutch = float(clutch_row.get("Clutch", "nan"))
            wpa    = float(clutch_row.get("WPA", "nan"))
            sd     = clutch_row.get("SD", "-")
            md     = clutch_row.get("MD", "-")
            clutch_desc = "기대 대비 클러치 상황에서 선전" if clutch > 0 else "기대 대비 클러치 상황에서 부진"
            lines += [
                f"- 평균 레버리지(pLI): {pli:.2f}  ·  WPA(승리 기여): {wpa:+.3f}",
                f"- Clutch 점수: {clutch:+.3f}  ({clutch_desc})",
                f"- 득점권 상황 등판(SD): {sd}회  ·  중요 상황 등판(MD): {md}회",
            ]
        except Exception:
            pass

    if save_row is not None:
        try:
            sv_n   = int(float(save_row.get("SV", 0)))
            bs_n   = int(float(save_row.get("BS", 0)))
            total  = sv_n + bs_n
            sv_era = save_row.get("ERA", "-")
            if total > 0:
                bs_rate = bs_n / total * 100
                lines.append(
                    f"- 세이브 상황: {sv_n}세이브 / {bs_n}블론 / 기회 {total}회 / 블론율 {bs_rate:.0f}% / ERA {sv_era}"
                )
        except Exception:
            pass

    meta = data["meta"]["pitchers"].get(player, {})
    sit  = meta.get("situation", "-")
    n_a  = meta.get("n_a", "-"); n_b = meta.get("n_b", "-")
    lines.append(f"- 모션 분석 케이스: {sit} (좋은 결과 {n_a}경기 / 나쁜 결과 {n_b}경기)")
    return lines or ["- 상황별 성적 데이터가 없습니다."]


def _player_kinematic_detail_lines(player: str) -> list[str]:
    pitcher_df = data["pitcher_ag"][data["pitcher_ag"]["player"] == player].copy()
    if pitcher_df.empty:
        return ["- 키네마틱 분석 데이터가 없습니다."]
    top = pitcher_df.reindex(
        pitcher_df["cohens_d"].abs().sort_values(ascending=False).index
    ).head(5)
    # 비교 상황 (A/B) 라벨 — 첫 행에서 추출
    sit_a = str(top.iloc[0].get("situation_a", "A"))
    sit_b = str(top.iloc[0].get("situation_b", "B"))
    lines: list[str] = []
    # 헤더 — 비전공자 친화 1줄 가이드
    lines.append(f"※ 같은 선수의 *{sit_a} 경기*와 *{sit_b} 경기*에서 폼이 얼마나 달랐는지 비교. 차이 클수록 *상황별 폼 변동*이 큼.")
    lines.append("   '차이 크기' = Cohen's d (0.2 작음 / 0.5 중간 / 0.8 큼 / 2.0+ 매우 큼).  '통계 명확성' = p-value (0.05 미만이면 우연 가능성 낮음).")
    lines.append("")
    lines = []
    for _, row in top.iterrows():
        metric = str(row.get("metric", ""))
        metric_name, description = _METRIC_INFO.get(metric, (metric, ""))
        d    = float(row["cohens_d"])
        p    = float(row["u_p"])
        diff = float(row["diff"])
        direction = "더 컸음" if diff > 0 else "더 작았음"
        d_label = _cohens_d_label(d)
        p_label = _sig_label(p)
        lines += [
            f"■ {metric_name}",
            f"   {sit_a} 경기에서 {abs(diff):.2f} {direction}  ·  차이 크기 {d_label} (d={d:.2f})  ·  {p_label} (p={p:.3f})",
        ]
        if description:
            lines.append(f"   → {description}")
    return lines


# ──────────────────────────────────────────────────
# OpenBiomechanics 레퍼런스 비교 (411 college~prosp 투수)
# ──────────────────────────────────────────────────

# 우리 metric → openbiomechanics 컬럼 매핑 + 단위 변환 계수 (poi 값에 곱함)
# 주의: 방송 영상(30fps, 시간 해상도 33ms) 기반이라 *시간 단위 metric*은 제외.
#       각도/각속도는 절대값 차이는 있을 수 있으나 percentile 위치는 유의미.
_OPENBIO_MAP: dict[str, tuple[str, float]] = {
    "hip_peak_dps":    ("max_pelvis_rotational_velo", 1.0),
    "trunk_peak_dps":  ("max_torso_rotational_velo",  1.0),
    "hip_3d_dps":      ("max_pelvis_rotational_velo", 1.0),
    "trunk_3d_dps":    ("max_torso_rotational_velo",  1.0),
    "hss_at_fp_deg":   ("rotation_hip_shoulder_separation_fp", 1.0),
    "hss_max_deg":     ("max_rotation_hip_shoulder_separation", 1.0),
    # "timing_diff_ms": 방송 33ms 해상도 vs force plate 2.8ms → 직접 비교 부적절, 제외
    # "trunk_hip_ratio": openbiomechanics에 직접 컬럼 없음
}


def _load_poi_metrics():
    if "_poi_cached" in _load_poi_metrics.__dict__:
        return _load_poi_metrics._poi_cached

    candidates = [
        DATA_DIR / "poi_metrics.csv",
        PROJECT_ROOT / "Notebooks" / "지소윤" / "baseball_kinematics" / "openbiomechanics" / "baseball_pitching" / "data" / "poi" / "poi_metrics.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                df = pd.read_csv(path)
                _load_poi_metrics._poi_cached = df
                return df
            except Exception:
                pass

    _load_poi_metrics._poi_cached = None
    return None


def _player_reference_comparison_lines(player: str) -> list[str]:
    """openbiomechanics 411명 투수 레퍼런스와 비교 — 비전공자 친화 출력."""
    poi = _load_poi_metrics()
    if poi is None or poi.empty:
        return ["- 비교 데이터를 불러올 수 없습니다."]
    pitcher_df = data["pitcher_ag"][data["pitcher_ag"]["player"] == player].copy()
    if pitcher_df.empty:
        return ["- 선수 동작 분석 데이터가 없습니다."]

    lines: list[str] = []
    lines.append("이 선수의 폼을 *동급 411명 투수 데이터셋(openbiomechanics)*과 비교했습니다. 상위 %는 강함, 하위 %는 약함.")
    lines.append("※ 우리 데이터는 방송 영상에서 추출한 2D→3D 추정값 — *절대 수치보다 상대 순위*로 해석. 시간 측정은 비교 제외.")
    lines.append("")

    lines = []
    high_metrics = []   # 상위권 (강점)
    low_metrics = []    # 하위권 (약점)

    for _, row in pitcher_df.iterrows():
        metric = str(row["metric"])
        if metric not in _OPENBIO_MAP:
            continue
        poi_col, scale = _OPENBIO_MAP[metric]
        if poi_col not in poi.columns:
            continue
        ref_vals = poi[poi_col].dropna() * scale
        if len(ref_vals) < 10:
            continue

        n_a, n_b = int(row.get("n_a", 0)), int(row.get("n_b", 0))
        n_total = max(1, n_a + n_b)
        player_mean = (float(row["a_mean"]) * n_a + float(row["b_mean"]) * n_b) / n_total

        ref_mean = float(ref_vals.mean())
        ref_std  = float(ref_vals.std())
        pct = float((ref_vals < player_mean).mean()) * 100

        # 비전공자 친화 라벨
        if pct >= 90:   pct_label, intuition = "상위 10%", "(최상위 수준)"
        elif pct >= 80: pct_label, intuition = "상위 20%", "(강한 편)"
        elif pct >= 60: pct_label, intuition = "상위 40%", "(평균보다 좋음)"
        elif pct >= 40: pct_label, intuition = "중위 (평균 근처)", "(평범)"
        elif pct >= 20: pct_label, intuition = "하위 40%", "(평균보다 약함)"
        elif pct >= 10: pct_label, intuition = "하위 20%", "(약한 편)"
        else:           pct_label, intuition = "하위 10%", "(최하위 수준)"

        label = str(row.get("label", metric))

        # 단위 표시
        if "각속도" in label:
            unit = " °/s"
        elif "°" in label or "FP" in label or "max" in label.lower():
            unit = "°"
        else:
            unit = ""

        # 2줄로 압축 — metric 1행 + 위치 1행
        lines += [
            f"■ {label} — 이 선수 {player_mean:.0f}{unit}  vs  411명 평균 {ref_mean:.0f}±{ref_std:.0f}{unit}",
            f"   → {pct_label} {intuition}",
        ]

        # 강점/약점 집계
        if pct >= 80:
            high_metrics.append((label, pct))
        elif pct <= 20:
            low_metrics.append((label, pct))

    if len(lines) <= 5:
        return ["- 비교 가능한 metric이 없습니다."]
    # 한 줄 요약 제거 — 선수별 패턴은 다음의 Action Priority 섹션이 종합 해석 담당
    return lines


def _player_action_priority_lines(player: str) -> list[str]:
    _ACTION_MAP: dict[str, dict] = {
        "Leiter": {
            "signal":      "약 (Cohen's d 대부분 < 0.5)",
            "coaching":    ["초구 스트라이크 확보율 점검", "구종 구성 다양화 (체인지업/커브 비중 재검토)"],
            "operational": ["불리한 카운트별 구종 운영 프로토콜 표준화", "좌/우 타자별 접근법 및 스플릿 점검"],
        },
        "Webb": {
            "signal":      "강 (Trunk/Hip ratio d=3.05, HSS @ FP d=2.16)",
            "coaching":    ["하체-상체 분리(HSS @ FP) 안정화 루틴 설계", "릴리스 전 몸통 회전 타이밍 교정"],
            "operational": ["연속 등판 제한 (3일 이내 재등판 주의)", "하이 레버리지 이닝 연속 배치 관리"],
        },
        "Garcia": {
            "signal":      "약 (Cohen's d 대부분 < 0.9, p > 0.25) — null finding 우세",
            "coaching":    ["현재 모션 분석상 명확한 폼 교정 대상 없음"],
            "operational": ["마무리 고정 여부 재검토", "좌타자 매치업 재설계", "연투 후 다음 등판 성과 추적"],
        },
        "Armstrong": {
            "signal":      "중간 (일부 지표 변동, 표본 소규모로 해석 제한)",
            "coaching":    ["워밍업 루틴 표준화 (짧은 등판 특성 고려)"],
            "operational": ["구위 하락 경기 다음 등판 패턴 확인", "특정 타순·상황 전용 기용 여부 판단"],
        },
        "Jackson": {
            "signal":      "약-중간 (사이드암보다 측면 기울기 오버핸드 패턴으로 재해석)",
            "coaching":    ["현재 투구 패턴 폼 교정보다 유지 권장"],
            "operational": ["플래툰 스플릿 기반 specialist 배치", "좌타자 상대 제한 운영", "전천후 기용 회피"],
        },
    }
    info = _ACTION_MAP.get(player)
    if not info:
        return ["- 해당 선수의 액션 우선순위 데이터가 없습니다."]
    lines = [f"모션 신호 강도: {info['signal']}", ""]
    lines.append("[코칭 가능 항목]")
    lines.extend(f"- {item}" for item in info["coaching"])
    lines.append("[운영·배치로 풀 항목]")
    lines.extend(f"- {item}" for item in info["operational"])
    return lines


def _team_bullpen_lines() -> list[str]:
    save_df   = _read_raw_csv("tex_2025_save_situation_splits.csv")
    clutch_df = _read_raw_csv("tex_clutch_pit.csv")
    if save_df.empty:
        return ["- 세이브 상황 데이터를 찾지 못했습니다."]
    lines: list[str] = []

    # 팀 전체 세이브 상황 요약
    team_mask = save_df["Name"].astype(str).str.contains("Team Total", case=False, na=False)
    if team_mask.any():
        r = save_df.loc[team_mask].iloc[0]
        sv = r.get("SV", "-"); bs = r.get("BS", "-"); era = r.get("ERA", "-")
        try:
            total   = int(float(sv)) + int(float(bs))
            bs_rate = int(float(bs)) / total * 100 if total > 0 else 0
            lines.append(
                f"- 팀 전체: {sv}세이브 / {bs}블론 / 총 {total}기회 / 블론율 {bs_rate:.0f}% / 상황 ERA {era}"
            )
            lines.append(
                f"  → BS {bs}회는 MLB 28위권 수준 — ERA 1위(실점 억제 최고)이면서도 잔차 -9승이 생긴"
                f" 핵심 이유가 바로 세이브 상황 붕괴입니다."
            )
        except Exception:
            lines.append(f"- 팀 SV {sv} / BS {bs}")

    # 개별 투수 WPA 조회
    def _wpa(name_key: str) -> str:
        if clutch_df.empty:
            return ""
        for col in ("NameASCII", "Name"):
            if col not in clutch_df.columns:
                continue
            mask = clutch_df[col].astype(str).str.contains(name_key, case=False, regex=False)
            if mask.any():
                w = clutch_df.loc[mask, "WPA"].values[0] if "WPA" in clutch_df.columns else float("nan")
                try:
                    return f" / WPA {float(w):+.3f}"
                except Exception:
                    return ""
        return ""

    lines.append("주요 투수별 세이브 상황:")
    key_pitchers = [("Garcia", "마무리"), ("Armstrong", "셋업"), ("Jackson", "계투"),
                    ("Martin", "계투"), ("Milner", "계투")]
    for kp, role in key_pitchers:
        mask = save_df["Name"].astype(str).str.contains(kp, case=False, na=False)
        if not mask.any():
            continue
        r    = save_df.loc[mask].iloc[0]
        sv   = r.get("SV", 0); bs_v = r.get("BS", 0); name = r.get("Name", kp)
        era  = r.get("ERA", "-")
        wpa  = _wpa(kp)
        try:
            total = int(float(sv)) + int(float(bs_v))
            if total > 0:
                bs_rate = int(float(bs_v)) / total * 100
                lines.append(
                    f"  - {name} ({role}): {sv}SV / {bs_v}BS / 블론율 {bs_rate:.0f}%"
                    f" / ERA {era}{wpa}"
                )
        except Exception:
            lines.append(f"  - {name} ({role}): SV {sv} / BS {bs_v}")

    # 핵심 구조적 사실 — 불펜 취약 이유 설명
    lines += [
        "",
        "불펜 붕괴 구조 분석:",
        "- Robert Garcia: BS 7회, WPA -0.93(팀 최하위) — 하이 레버리지 등판마다 기대 이하 성과가 누적됐습니다.",
        "- Hoby Milner: 하이 레버리지 상황 피슬래시 .333/.362/.448 — 중요 장면에서 오히려 피타율이 올라갔습니다.",
        "- Chris Martin: 시즌 중 IL 3회·55경기 결장 — 셋업 역할 공백이 중후반 불펜 운용을 왜곡했습니다.",
        "- 트레이드 데드라인: Phil Maton·Danny Coulombe 영입으로 보강을 시도했으나 시즌 후반 흐름 반전에는 부족했습니다.",
    ]
    return lines


def _team_clutch_lines() -> list[str]:
    clutch_df = _read_raw_csv("tex_clutch_pit.csv")
    if clutch_df.empty:
        return ["- 클러치 데이터를 찾지 못했습니다."]
    df = clutch_df.copy()
    df["_wpa"]    = pd.to_numeric(df.get("WPA",    pd.Series(dtype=float)), errors="coerce")
    df["_clutch"] = pd.to_numeric(df.get("Clutch", pd.Series(dtype=float)), errors="coerce")
    lines: list[str] = []
    lines.append("WPA 상위 기여 투수 (실제 승리 기여도 높음):")
    for _, row in df.nlargest(3, "_wpa").iterrows():
        name = row.get("Name", "-"); wpa = float(row["_wpa"])
        lines.append(f"- {name}: WPA {wpa:+.3f}")
    lines.append("Clutch 점수 하위 투수 (하이 레버리지에서 기대보다 부진):")
    for _, row in df.nsmallest(3, "_clutch").iterrows():
        name = row.get("Name", "-"); c = float(row["_clutch"])
        lines.append(f"- {name}: Clutch {c:+.3f}")
    lines.append("- Clutch 점수 음수 = 하이 레버리지에서 기대 대비 득점 허용이 많음을 의미합니다.")
    return lines


def _team_inning_lines() -> list[str]:
    df = _read_raw_csv("tex_2025_pitching_inning_splits.csv")
    if df.empty:
        return ["- 이닝별 성적 데이터를 찾지 못했습니다."]
    df["_Split"] = df["Split"].astype(str).str.strip()

    _INNING_LABELS = {
        "1st inning": "1회",
        "2nd inning": "2회",
        "3rd inning": "3회",
        "4th inning": "4회",
        "5th inning": "5회",
        "6th inning": "6회",
        "7th inning": "7회",
        "8th inning": "8회",
        "9th inning": "9회",
        "Ext inning":  "연장",
    }

    lines: list[str] = []

    # 회별 세부 ERA 테이블
    lines.append("회별 ERA (피OPS):")
    era_map: dict[str, float] = {}
    for key, label in _INNING_LABELS.items():
        row = df[df["_Split"] == key]
        if row.empty:
            continue
        r = row.iloc[0]
        try:
            era = float(r["ERA"]); ops = float(r["OPS"])
            era_map[key] = era
            lines.append(f"  {label}: ERA {era:.2f} / 피OPS {ops:.3f}")
        except Exception:
            pass

    # 핵심 패턴 해석
    lines.append("")
    e8  = era_map.get("8th inning", float("nan"))
    e9  = era_map.get("9th inning", float("nan"))
    ext = era_map.get("Ext inning",  float("nan"))
    e13 = era_map.get("1st inning",  float("nan"))

    if not any(map(pd.isna, [e8, e9])) and e9 > e8:
        diff89 = e9 - e8
        lines.append(
            f"- 8회 ERA {e8:.2f} → 9회 ERA {e9:.2f} (차이 +{diff89:.2f}): "
            f"8회까지 효율적으로 막다가 9회에서 붕괴하는 패턴 — 전문 마무리 부재의 직접 결과입니다."
        )
    if not pd.isna(ext):
        lines.append(
            f"- 연장 ERA {ext:.2f}: 연장 상황에서 실점이 급증해 접전 경기를 승리로 전환하는 데 실패했습니다."
        )

    # 구간별 ERA 비교
    lines.append("")
    lines.append("구간별 ERA:")
    for split, label in [("Innings 1-3", "선발 초반(1-3회)"),
                         ("Innings 4-6", "선발 중반(4-6회)"),
                         ("Innings 7-9", "불펜(7-9회)")]:
        row = df[df["_Split"] == split]
        if row.empty:
            continue
        r = row.iloc[0]
        try:
            era = float(r["ERA"]); ops = float(r["OPS"])
            lines.append(f"  {label}: ERA {era:.2f} / 피OPS {ops:.3f}")
        except Exception:
            pass

    return lines


def _team_pitching_staff_lines() -> list[str]:
    """선발 로테이션 + 불펜 주요 성적 요약 및 전문 마무리 부재 분석."""
    pit_df  = _read_raw_csv("texas_pitchers_2025.csv")
    save_df = _read_raw_csv("tex_2025_save_situation_splits.csv")
    if pit_df.empty:
        return ["- 투수 성적 데이터를 찾지 못했습니다."]

    for c in ("IP", "ERA", "FIP", "WAR", "GS", "G", "SV", "BS", "K/9", "BB/9"):
        if c in pit_df.columns:
            pit_df[c] = pd.to_numeric(pit_df[c], errors="coerce")

    starters  = pit_df[pit_df["GS"] >= 5].sort_values("WAR", ascending=False)
    relievers = pit_df[pit_df["GS"] < 5].sort_values("WAR", ascending=False)

    lines: list[str] = []

    # ── 선발 로테이션 ─────────────────────────────────────
    lines.append("선발 로테이션 성적:")
    for _, r in starters.iterrows():
        name = str(r.get("Name", "-"))
        g    = int(r.get("G",   0))
        gs   = int(r.get("GS",  0))
        ip   = float(r.get("IP",  float("nan")))
        era  = float(r.get("ERA", float("nan")))
        fip  = float(r.get("FIP", float("nan")))
        war  = float(r.get("WAR", float("nan")))
        ip_s  = f"{ip:.1f}" if not pd.isna(ip)  else "-"
        era_s = f"{era:.2f}" if not pd.isna(era) else "-"
        fip_s = f"{fip:.2f}" if not pd.isna(fip) else "-"
        war_s = f"{war:.1f}" if not pd.isna(war) else "-"

        # 해석 태그 (FIP와 ERA 모두 참고)
        if not pd.isna(era) and not pd.isna(fip):
            ref_era = max(era, fip)  # FIP가 높으면 실질 구위로 보정
            if ref_era < 3.20:
                tag = "에이스급"
            elif ref_era < 4.00:
                tag = "안정적"
            elif ref_era < 5.00:
                tag = "보완 필요"
            else:
                tag = "교체 검토"
        else:
            tag = ""
        # GS 비율이 낮으면 겸용 표시
        role_tag = "선발/불펜 겸용" if g > 0 and gs / g < 0.6 else ""
        combined_tag = " / ".join(filter(None, [tag, role_tag]))
        tag_s = f" [{combined_tag}]" if combined_tag else ""
        lines.append(
            f"  - {name}: {gs}선발 / {ip_s}이닝 / ERA {era_s} / FIP {fip_s} / WAR {war_s}{tag_s}"
        )

    # ── 불펜 핵심 인원 ────────────────────────────────────
    lines.append("")
    lines.append("불펜 핵심 투수:")
    for _, r in relievers.head(8).iterrows():
        name = str(r.get("Name", "-"))
        g    = int(r.get("G",  0))
        ip   = float(r.get("IP",  float("nan")))
        era  = float(r.get("ERA", float("nan")))
        sv   = int(r.get("SV", 0))
        bs   = int(r.get("BS", 0))
        war  = float(r.get("WAR", float("nan")))
        ip_s  = f"{ip:.1f}" if not pd.isna(ip)  else "-"
        era_s = f"{era:.2f}" if not pd.isna(era) else "-"
        war_s = f"{war:.1f}" if not pd.isna(war) else "-"
        sv_bs = f" / {sv}SV {bs}BS" if (sv + bs) > 0 else ""
        lines.append(
            f"  - {name}: {g}G / {ip_s}이닝 / ERA {era_s}{sv_bs} / WAR {war_s}"
        )

    # ── 전문 마무리 부재 진단 ─────────────────────────────
    lines.append("")
    lines.append("전문 마무리 부재 진단:")

    # 세이브 분산도 계산
    closers = [("Jackson", "Luke Jackson"), ("Armstrong", "Shawn Armstrong"), ("Garcia", "Robert Garcia")]
    closer_sv: list[tuple[str, int, int, str]] = []
    for key, display in closers:
        mask = pit_df["Name"].astype(str).str.contains(key, case=False, na=False)
        if not mask.any():
            continue
        r  = pit_df.loc[mask].iloc[0]
        sv = int(r.get("SV", 0)); bs = int(r.get("BS", 0))
        era_v = r.get("ERA", float("nan"))
        era_s = f"{float(era_v):.2f}" if not pd.isna(era_v) else "-"
        closer_sv.append((display, sv, bs, era_s))

    total_sv = sum(x[1] for x in closer_sv)
    total_bs_team = 29  # 팀 총 BS (세이브 상황 데이터 기준)

    for display, sv, bs, era_s in closer_sv:
        total_opp = sv + bs
        bs_rate = bs / total_opp * 100 if total_opp > 0 else 0
        sv_share = sv / total_sv * 100 if total_sv > 0 else 0
        lines.append(
            f"  - {display}: {sv}세이브 (전체의 {sv_share:.0f}%) / {bs}블론 / 블론율 {bs_rate:.0f}% / 시즌 ERA {era_s}"
        )

    lines += [
        f"  → Jackson·Armstrong·Garcia 세 명이 세이브를 {total_sv}개 나눠 가졌습니다.",
        f"    단일 마무리가 정착하지 못한 상태에서 팀 전체 블론 {total_bs_team}회가 누적됐고,",
        f"    '잡은 리드를 지키지 못하는' 패턴이 -9.06승 잔차의 가장 큰 구조적 원인입니다.",
    ]

    return lines


def _team_decision_matrix_lines() -> list[str]:
    matrix = [
        ("Leiter",    "선발",   "약 (d<0.5)",        "운영",   "구종 선택·카운트 운영"),
        ("Webb",      "선발",   "강 (d=2~3)",         "코칭",   "하체-상체 분리, 타이밍"),
        ("Garcia",    "마무리", "약 (d<0.9, null)",   "운영",   "매치업·연투 관리"),
        ("Armstrong", "셋업",   "중간 (d~0.7~1.0)",  "운영",   "워밍업 루틴·등판 빈도"),
        ("Jackson",   "계투",   "약-중 (패턴 재해석)", "운영",  "플래툰 전용 배치"),
    ]
    lines = ["선수 / 역할 / 모션 신호 강도 / 우선 조치 / 핵심 포인트", "─" * 58]
    matrix[0], matrix[1] = matrix[1], matrix[0]
    for player, role, signal, priority, point in matrix:
        lines.append(f"{player} ({role}) | 모션: {signal} | {priority} 우선 | {point}")
    return lines


def _team_batting_lines() -> list[str]:
    """팀 타격 지표 요약 + wRC+ 기준 선수 현황."""
    wrc_df  = _read_raw_csv("tex_wrc+.csv")
    bat_df  = _read_raw_csv("batting_stats_2025_all.csv")
    mlb_df  = _read_raw_csv("mlb_team_seasons.csv")
    lines: list[str] = []

    # ── 팀 OPS·득점 ──────────────────────────────────────
    if not mlb_df.empty and "year" in mlb_df.columns:
        mlb25 = mlb_df[pd.to_numeric(mlb_df["year"], errors="coerce") == 2025].copy()
        tex   = mlb25[mlb25["team"].astype(str).str.contains("Texas|Rangers", case=False, na=False)]
        if not tex.empty and not mlb25.empty:
            r        = tex.iloc[0]
            rs       = int(float(r.get("RS", 0)))
            ops_v    = float(r.get("OPS", float("nan")))
            sb       = int(float(r.get("SB", 0)))
            ops_s    = pd.to_numeric(mlb25["OPS"], errors="coerce")
            ops_rank = int((ops_s > ops_v).sum() + 1)
            n        = int(ops_s.dropna().count())
            if ops_rank > 20:
                ops_interp = f"리그 하위권({ops_rank}/{n}위) → 팀 전체 득점 생산력이 잔차를 키운 주요 원인"
            elif ops_rank > 15:
                ops_interp = f"리그 평균 이하({ops_rank}/{n}위) → 타선 전체 득점 생산력 제한"
            else:
                ops_interp = f"리그 {ops_rank}/{n}위"
            lines.append(
                f"- 팀 득점: {rs}점  ·  팀 OPS: {ops_v:.3f} ({ops_interp})  ·  도루: {sb}개"
            )

    # ── wRC+ 선수별 현황 ──────────────────────────────────
    if not wrc_df.empty and "wRC+" in wrc_df.columns:
        wrc_df["_w"] = pd.to_numeric(wrc_df["wRC+"], errors="coerce")
        top5 = wrc_df.dropna(subset=["_w"]).sort_values("_w", ascending=False).head(5)
        lines.append("타선 주요 선수 (wRC+ 기준):")
        tex_bat = pd.DataFrame()
        if not bat_df.empty and "Tm" in bat_df.columns:
            tex_bat = bat_df[bat_df["Tm"].astype(str).str.contains("Texas|Rangers", case=False, na=False)]
        for _, row in top5.iterrows():
            name  = str(row.get("Name", "-"))
            wrc_v = float(row["_w"])
            extra = ""
            if not tex_bat.empty:
                last  = name.split()[-1]
                pmask = tex_bat["Name"].astype(str).str.contains(last, case=False, regex=False)
                if pmask.any():
                    pr = tex_bat.loc[pmask].iloc[0]
                    g  = int(float(pr.get("G", 0)))
                    hr = int(float(pr.get("HR", 0)))
                    op = float(pr.get("OPS", float("nan")))
                    extra = f" | {g}G / {hr}HR / OPS {op:.3f}"
            lines.append(f"  - {name}: wRC+ {wrc_v:.0f}{extra}")

        # 타선 깊이 평가
        wrc_vals = top5["_w"].tolist()
        if len(wrc_vals) >= 2:
            leader = float(wrc_vals[0])
            second = float(wrc_vals[1])
            fifth  = float(wrc_vals[-1])
            if leader >= 130 and second < 125:
                lines.append(
                    f"  → Seager(wRC+ {leader:.0f})가 타선을 이끌지만 2~5위({second:.0f}~{fifth:.0f})는"
                    f" 리그 평균 수준에 불과 — Seager 외 확실한 중심타선이 없어 부상·부진 시 타선 전체가 빠르게 붕괴됩니다."
                )

    return lines or ["- 타격 데이터를 찾지 못했습니다."]


def _team_defense_lines() -> list[str]:
    """팀 수비 관련 지표 (투수-수비 복합 프록시) — 순위 기반 해석."""
    teams  = _read_raw_csv("mlb_teams_2025_pitching.csv")
    mlb_df = _read_raw_csv("mlb_team_seasons.csv")
    lines: list[str] = []

    if not teams.empty and "Team" in teams.columns:
        tex = teams[teams["Team"].astype(str).str.upper() == "TEX"]
        if not tex.empty:
            r = tex.iloc[0]

            def _fmt_val(v: float, series: pd.Series) -> str:
                return f"{v:.1f}%" if series.dropna().max() > 1.0 else f"{v:.3f}"

            def _rank_of(col: str, higher_better: bool) -> tuple[float, int, int] | None:
                if col not in teams.columns:
                    return None
                val = pd.to_numeric(pd.Series([r[col]]), errors="coerce").iloc[0]
                if pd.isna(val):
                    return None
                all_v = pd.to_numeric(teams[col], errors="coerce").dropna()
                rank = int((all_v > val).sum() + 1) if higher_better else int((all_v < val).sum() + 1)
                return val, rank, len(all_v)

            res = _rank_of("BABIP", False)
            if res:
                val, rank, n = res
                interp = ("리그 최상위 수준의 안타 억제 → 수비·피칭이 맞물려 잘 작동" if rank <= 5
                          else "리그 평균 수준의 안타 허용" if rank <= 15
                          else "안타 허용 억제 측면에서 약점")
                lines.append(f"- BABIP 허용: {val:.3f} (MLB {rank}/{n}위) → {interp}")

            res = _rank_of("Hard%", False)
            if res:
                val, rank, n = res
                all_v = pd.to_numeric(teams["Hard%"], errors="coerce").dropna()
                display = _fmt_val(val, all_v)
                interp = ("강한 타구를 잘 억제 → 투수 구위·수비 합산 효과 우수" if rank <= 10
                          else "강한 타구 허용이 평균보다 많아 장타 허용 위험 내재" if rank > 20
                          else "강한 타구 허용 리그 평균 수준")
                lines.append(f"- 강한 타구 허용률(Hard%): {display} (MLB {rank}/{n}위) → {interp}")

            res = _rank_of("GB%", True)
            if res:
                val, rank, n = res
                all_v = pd.to_numeric(teams["GB%"], errors="coerce").dropna()
                display = _fmt_val(val, all_v)
                interp = ("땅볼 유도 능력 우수 → 장타 억제에 실질적으로 유리" if rank <= 10
                          else "땅볼 비율 평균 이하 → 장타 억제 측면에서 강점이 아님" if rank > 20
                          else "땅볼 비율 리그 평균 수준")
                lines.append(f"- 땅볼 비율(GB%): {display} (MLB {rank}/{n}위) → {interp}")

            res = _rank_of("HR/FB", False)
            if res:
                val, rank, n = res
                all_v = pd.to_numeric(teams["HR/FB"], errors="coerce").dropna()
                display = _fmt_val(val, all_v)
                interp = ("플라이볼 대비 홈런 허용이 적음 → 구장 환경·구위 합산 억제 효과" if rank <= 10
                          else "홈런 허용 비율이 높아 장타 허용 위험" if rank > 20
                          else "홈런 허용 비율 리그 평균 수준")
                lines.append(f"- 홈런/플라이볼(HR/FB): {display} (MLB {rank}/{n}위) → {interp}")

    if not mlb_df.empty and "year" in mlb_df.columns:
        mlb25 = mlb_df[pd.to_numeric(mlb_df["year"], errors="coerce") == 2025].copy()
        tex   = mlb25[mlb25["team"].astype(str).str.contains("Texas|Rangers", case=False, na=False)]
        if not tex.empty:
            r       = tex.iloc[0]
            onerun  = float(r.get("onerun_wp", float("nan")))
            onerun_n = int(float(r.get("onerun_n", 0)))
            if not pd.isna(onerun):
                interp = ("접전에서도 승리를 챙기는 패턴 → 잔차 완충 효과 일부 있음" if onerun >= 0.500
                          else "접전에서 무너지는 패턴 → 블론 세이브와 맞물려 잔차를 확대시킨 직접 원인")
                lines.append(f"- 1점차 경기 승률: {onerun:.3f} ({onerun_n}경기) → {interp}")

    return lines or ["- 수비 관련 데이터를 찾지 못했습니다."]


def _team_monthly_lines() -> list[str]:
    """월별 팀 성적 (W/L/RS/RA/승률)."""
    df = _read_raw_csv("texas_2025_game_log.csv")
    if df.empty or "Date" not in df.columns:
        return ["- 경기 기록 데이터를 찾지 못했습니다."]

    import re
    _MONTH_ORDER = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    def _parse_month(d: str) -> str:
        m = re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)", str(d))
        return m.group(1) if m else "Unknown"

    df = df.copy()
    df["_month"] = df["Date"].apply(_parse_month)
    df["_win"]   = df["W/L"].astype(str).str.startswith("W").astype(int)
    df["_loss"]  = df["W/L"].astype(str).str.startswith("L").astype(int)
    df["_rs"]    = pd.to_numeric(df["R"],  errors="coerce")
    df["_ra"]    = pd.to_numeric(df["RA"], errors="coerce")

    monthly = (
        df.groupby("_month")
        .agg(G=("_win", "count"), W=("_win", "sum"), L=("_loss", "sum"),
             RS=("_rs", "sum"), RA=("_ra", "sum"))
        .reset_index()
    )
    monthly["_ord"] = monthly["_month"].apply(
        lambda x: _MONTH_ORDER.index(x) if x in _MONTH_ORDER else 99
    )
    monthly = monthly.sort_values("_ord")

    lines: list[str] = []
    for _, row in monthly.iterrows():
        mon  = row["_month"]; g = int(row["G"]); w = int(row["W"]); l = int(row["L"])
        rs   = int(row["RS"]); ra = int(row["RA"])
        wpct = w / g if g > 0 else 0
        diff = rs - ra
        sign = "+" if diff >= 0 else ""
        lines.append(
            f"- {mon}: {w}승 {l}패 / 승률 {wpct:.3f} / 득점 {rs} / 실점 {ra} / 득실차 {sign}{diff}"
        )

    # ── 월별 패턴 해석 ────────────────────────────────────
    if not monthly.empty:
        monthly["_diff"] = monthly["RS"] - monthly["RA"]
        monthly["_wpct"] = monthly.apply(lambda r: r["W"] / r["G"] if r["G"] > 0 else 0, axis=1)

        best_w  = monthly.loc[monthly["W"].idxmax()]
        worst_w = monthly.loc[monthly["W"].idxmin()]
        best_d  = monthly.loc[monthly["_diff"].idxmax()]

        lines.append("")
        lines.append("월별 흐름 해석:")

        # Peak month
        bm  = str(best_w["_month"]); bw = int(best_w["W"]); bl = int(best_w["G"]) - bw
        bd  = int(best_d["_diff"]); bdm = str(best_d["_month"])
        lines.append(
            f"- {bm}이 {bw}승 {bl}패로 시즌 최고 성적. 득실차 기준 최고월은 {bdm}(+{bd}점)."
        )

        # Late-season collapse
        sep = monthly[monthly["_month"] == "Sep"]
        aug = monthly[monthly["_month"] == "Aug"]
        if not sep.empty:
            sw  = int(sep.iloc[0]["W"]); sl = int(sep.iloc[0]["G"]) - sw
            sd  = int(sep.iloc[0]["_diff"]); sign_sd = "+" if sd >= 0 else ""
            lines.append(
                f"- 9월 {sw}승 {sl}패(득실차 {sign_sd}{sd}) — 플레이오프 경합 마지막 달에 승률이 급락했습니다."
            )
            if not aug.empty:
                aw  = int(aug.iloc[0]["W"]); al = int(aug.iloc[0]["G"]) - aw
                ad  = int(aug.iloc[0]["_diff"]); sign_ad = "+" if ad >= 0 else ""
                lines.append(
                    f"- 8월({aw}승 {al}패, {sign_ad}{ad}) → 9월로의 성적 하락은 Garcia 블론 집중 구간·Martin IL 시기와"
                    f" 겹치며 불펜 붕괴가 팀 성적 하락의 직접 원인으로 작용했습니다."
                )

    return lines


def _team_absence_lines() -> list[str]:
    """주요 선수 결장 현황 (타자 PA 80↑·경기 120↓, 투수 별도 포함)."""
    bat_df  = _read_raw_csv("batting_stats_2025_all.csv")
    ros_df  = _read_raw_csv("rangers_roster_2025.csv")
    pit_df  = _read_raw_csv("texas_pitchers_2025.csv")
    TOTAL_GAMES = 162
    lines: list[str] = []

    # ── 타자 결장 ─────────────────────────────────────────
    batter_names: set[str] = set()
    if not ros_df.empty and "type" in ros_df.columns and "name" in ros_df.columns:
        batter_names = set(
            ros_df[ros_df["type"] == "batter"]["name"].astype(str).str.strip().tolist()
        )

    if not bat_df.empty and "Tm" in bat_df.columns:
        tex = bat_df[bat_df["Tm"].astype(str).str.contains("Texas|Rangers", case=False, na=False)].copy()
        tex["_g"]  = pd.to_numeric(tex["G"],  errors="coerce")
        tex["_pa"] = pd.to_numeric(tex["PA"], errors="coerce")

        candidates = tex[(tex["_pa"] >= 80) & (tex["_g"] < 120)].copy()
        if batter_names:
            def _is_batter(name: str) -> bool:
                last = name.split()[-1]
                return any(last.lower() in bn.lower() for bn in batter_names)
            candidates = candidates[candidates["Name"].astype(str).apply(_is_batter)]

        candidates["_missed"] = TOTAL_GAMES - candidates["_g"]
        candidates = candidates[candidates["_missed"] >= 30].sort_values("_missed", ascending=False)

        for _, row in candidates.head(6).iterrows():
            name   = str(row.get("Name", "-"))
            g      = int(row["_g"])
            missed = int(row["_missed"])
            pa     = int(row["_pa"])
            ops    = float(row.get("OPS", float("nan")))
            ops_str = f" / OPS {ops:.3f}" if not pd.isna(ops) else ""
            lines.append(f"  - {name} (타자): {g}경기 출전 (PA {pa}) / 추정 결장 {missed}경기{ops_str}")

    # ── 선발투수 결장·제한 등판 ───────────────────────────
    if not pit_df.empty and "Name" in pit_df.columns:
        pit_df["_gs"] = pd.to_numeric(
            pit_df.get("GS", pit_df.get("G", pd.Series(dtype=float))), errors="coerce"
        )
        key_starters = [
            ("deGrom",    "Jacob deGrom",    "재활 복귀 후 이닝 제한 운용"),
        ]
        for key, display, note in key_starters:
            mask = pit_df["Name"].astype(str).str.contains(key, case=False, na=False)
            if not mask.any():
                continue
            r  = pit_df.loc[mask].iloc[0]
            gs = r.get("_gs", float("nan"))
            if pd.isna(gs):
                continue
            gs = int(gs)
            if gs < 30:
                lines.append(f"  - {display} (선발투수): {gs}경기 등판 — {note}")

    if not lines:
        return []

    header = ["- 주요 선수 결장·이탈 현황 (162경기 기준):"]
    footer = ["- 결장 수치는 출전 경기 역산 추정치이며, 부상 외 이유(선발 제외 등)도 포함될 수 있습니다."]
    return header + lines + footer


def _team_closer_transition_lines() -> list[str]:
    """마무리 투수 역할 변화 분석 (게임 로그 세이브 컬럼 기반)."""
    import re as _re

    game_df = _read_raw_csv("texas_2025_game_log.csv")
    save_df = _read_raw_csv("tex_2025_save_situation_splits.csv")
    if game_df.empty or "Date" not in game_df.columns:
        return ["- 게임 로그 데이터를 찾지 못했습니다."]

    _MONTH_ORDER = ["Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct"]

    def _parse_month(d: str) -> str:
        m = _re.search(r"(Mar|Apr|May|Jun|Jul|Aug|Sep|Oct)", str(d))
        return m.group(1) if m else "Unknown"

    df = game_df.copy()
    df["_month"]    = df["Date"].apply(_parse_month)
    df["_win"]      = df["W/L"].astype(str).str.startswith("W")
    df["_has_save"] = df["Save"].notna() & (df["Save"].astype(str).str.strip() != "")

    # TEX 세이브 = 팀이 이긴 경기에서 기록된 세이브
    tex_sv = df[df["_win"] & df["_has_save"]].copy()

    lines: list[str] = []

    # ── 월별 주요 마무리 투수 ─────────────────────────────
    lines.append("월별 마무리 투수 사용 현황 (팀 승리 세이브 기준):")
    phase_summary: list[str] = []
    prev_closer = ""
    for mon in _MONTH_ORDER:
        sub = tex_sv[tex_sv["_month"] == mon]
        if sub.empty:
            continue
        top = sub["Save"].value_counts()
        top1_name  = top.index[0]
        top1_count = int(top.iloc[0])
        total      = len(sub)
        others = [f"{k}({v})" for k, v in top.items() if k != top1_name]
        others_str = "  기타: " + ", ".join(others) if others else ""
        lines.append(
            f"  {mon}: 주 마무리 {top1_name} ({top1_count}/{total}회){others_str}"
        )
        if top1_name != prev_closer and prev_closer:
            phase_summary.append(f"{mon}에 {prev_closer} → {top1_name} 교체")
        prev_closer = top1_name

    # ── 교체 시점 요약 ────────────────────────────────────
    if phase_summary:
        lines.append("")
        lines.append("마무리 역할 변화 시점:")
        for s in phase_summary:
            lines.append(f"  - {s}")

    # ── 투수별 세이브 상황 성적 (BS 포함) ────────────────
    lines.append("")
    lines.append("주요 마무리 후보 세이브 상황 성적:")
    if not save_df.empty and "Name" in save_df.columns:
        key_closers = ["Garcia", "Armstrong", "Jackson"]
        for kp in key_closers:
            mask = save_df["Name"].astype(str).str.contains(kp, case=False, na=False)
            if not mask.any():
                continue
            r    = save_df.loc[mask].iloc[0]
            name = str(r.get("Name", kp))
            sv   = int(float(r.get("SV", 0)))
            bs   = int(float(r.get("BS", 0)))
            era  = r.get("ERA", "-")
            total = sv + bs
            bs_rate = bs / total * 100 if total > 0 else 0
            lines.append(
                f"  - {name}: {sv}SV / {bs}BS / 블론율 {bs_rate:.0f}%"
                f" (세이브 상황 ERA {era})"
            )

    lines.append("")
    lines.append(
        "- 역할 변화 해석: Jackson(초반 마무리) → 5월 분산 운용 → Garcia(6-7월 주 마무리) "
        "→ Armstrong(8-9월 사실상 마무리)으로 시즌 내 세 차례 교체가 발생했습니다."
    )
    lines.append(
        "- Garcia의 세이브 상황 블론율 44%(7BS)와 ERA 5.04는 7월 이후 역할 교체의 직접 원인으로 보입니다."
    )
    lines.append(
        "- Armstrong은 낮은 블론율(25%)로 마무리 전환 이후 상대적으로 안정적이었으나, "
        "시즌 초반부터 마무리로 운영됐다면 블론 세이브 총량을 줄일 수 있었을지 검토 가치가 있습니다."
    )
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
        "2025 Texas Rangers의 실제 81승과 피타고리안 기대 승수 90.06승 사이의 -9.06승 잔차를 설명하기 위한 선수별 보조 보고서입니다.",
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
        "2025 Texas Rangers는 실제 81승, 피타고리안 기대 승수 90.06승으로 -9.06승 잔차를 기록했습니다.",
        "본 대시보드는 이 차이를 설명하기 위해 경기력 분석, 선수 분석, 하이 레버리지 부진 대표 케이스 동작 분석, 시뮬레이션 의사결정 후보 비교를 순서대로 연결합니다.",
        "",
        "Analysis Flow",
        "1. 경기력 분석: 득실과 실제 승패가 어긋난 구간, 1점차/연장/세이브 상황 등 잔차가 커진 경기 맥락을 확인합니다.",
        "2. 선수 분석: 부상, 타격/투수 지표, 선수별 projection 변화가 팀 승수에 미친 조건을 분리합니다.",
        "3. 하이 레버리지 부진 대표 케이스 동작 분석: 코칭으로 고칠 수 있는 영역과 운영·보강이 필요한 영역을 3D 분석으로 구분합니다.",
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
        "하이 레버리지 부진 케이스 동작 분석은 9승 차이의 전체 원인이 아니라, 선수·운영 분석을 더 구체화하는 보조 근거입니다.",
        "Webb처럼 코칭 가능한 폼 분기가 있는 선수와 Garcia처럼 모션 외 요인을 우선 검토해야 하는 선수를 분리하는 것이 핵심입니다.",
        "최종 판단은 시뮬레이션의 의사결정 후보 순위표를 중심으로, 승수 개선 폭과 예측 안정성, 선수별 전망, 하이 레버리지 부진 케이스 동작 분석 근거를 함께 종합해야 합니다.",
    ])
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, "TEX 2025 Team Summary Report", lines)
    return buffer.getvalue()


# ── Data loading ──────────────────────────────────────────────
def build_player_report_pdf(player: str) -> bytes:
    info    = data["meta"]["pitchers"].get(player, {})
    finding = REPORT_FINDINGS.get(player, {})
    raw_row = _match_pitcher_row(player)
    display_name = str(raw_row["Name"]) if raw_row is not None and "Name" in raw_row.index else player
    lines = [
        "Purpose",
        "2025 Texas Rangers는 81승을 했습니다. 하지만 *득점·실점만 보면 90승*을 했어야 합니다 (피타고리안 기댓값).",
        "이 9승의 차이를 *어디서 잃었는지* 알아내려고, 결정적 순간(연장전·1점차·만루 등)에서 부진했던 투수를 골라",
        "성적·상황별 결과·3D 폼 분석을 합쳐 *코칭으로 풀 부분*과 *운영·배치로 풀 부분*을 구분합니다.",
        "",
        "Player Info",
        f"선수: {display_name}",
        f"역할: {finding.get('role', info.get('role', '-'))}",
        f"분석 비교: {info.get('situation', '-')}  ·  표본: A 상황 {info.get('n_a', '-')}경기 / B 상황 {info.get('n_b', '-')}경기",
        "",
        "Season Stats",
        *_player_season_stats_lines(player),
        "",
        "Team / League Context",
        *_player_percentile_lines(player),
        "",
        "하이 레버리지 & Clutch",
        *_player_situation_lines(player),
        "",
        "Motion Finding",
        finding.get("summary", "-"),
        "",
        "Kinematic Analysis Detail",
        *_player_kinematic_detail_lines(player),
        "",
        "OpenBiomechanics Reference",
        *_player_reference_comparison_lines(player),
        "",
        "Action Priority",
        *_player_action_priority_lines(player),
        "",
        "Recommendation",
        finding.get("recommendation", "-"),
        "",
        "Interpretation Note",
        "이 보고서는 모션 분석을 잔차 분석의 하위 근거로 사용합니다.",
        "단일 선수의 폼만으로 -9.06승 전체를 설명하지 않고, 경기 운영·Simulation 의사결정 후보와 함께 해석해야 합니다.",
        "- 코칭 가능 신호가 명확한 선수(Webb)와 운영·배치로 풀어야 할 선수(Garcia 등)를 분리하는 것이 핵심입니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        last_page = _add_pdf_page(pdf, f"TEX 2025 Player Report — {display_name}", lines)
        _add_player_radar_page(pdf, player, page_no=last_page + 1)
    return buffer.getvalue()


def build_team_report_pdf() -> bytes:
    lines = [
        "Purpose",
        "2025 Texas Rangers는 실제 81승, 피타고리안 기대 승수 90.06승으로 -9.06승의 잔차를 기록했습니다.",
        "팀 전체 리포트는 잔차 요약, 타격·수비 지표, 투수 지표 강약점,",
        "월별 성적 흐름, 주요 선수 결장 현황, 불펜 운영, 클러치·이닝 분석, 최종 결론을 한 흐름으로 정리합니다.",
        "",
        "Residual Summary",
        "Actual Wins: 81.0  ·  피타고리안 Expected: 90.06  ·  Residual: -9.06 wins",
        "- 득실점 기반 기대 승수보다 실제 승수가 약 9승 낮았습니다.",
        "- 이 차이는 전력 약세만이 아니라 접전 운영·세이브 실패·승패 타이밍의 누적 결과입니다.",
        "",
        "Batting Stats",
        *_team_batting_lines(),
        "",
        "Defense & Contact Suppression",
        *_team_defense_lines(),
        "",
        "Team Strength / Weakness",
        *_team_strength_lines(),
        "",
        "Pitching Staff Overview",
        *_team_pitching_staff_lines(),
        "",
        "Pitching Metric Rank",
        *_team_metric_lines(),
        "",
        "Monthly Record",
        *_team_monthly_lines(),
        "",
    ]
    _absence = _team_absence_lines()
    if _absence:
        lines += ["Key Player Absences", *_absence, ""]
    lines += [
        "Closer Role Transition",
        *_team_closer_transition_lines(),
        "",
        "Inning ERA by Role",
        *_team_inning_lines(),
        "",
        "Bullpen Save Situation",
        *_team_bullpen_lines(),
        "",
        "Clutch Performance",
        *_team_clutch_lines(),
        "",
    ]

    # ── 결론: ERA·FIP 동적 조회 ────────────────────────────
    _teams_raw = _read_raw_csv("mlb_teams_2025_pitching.csv")
    _era_str = "리그 최저"; _fip_str = "-"; _era_rank = 1; _fip_rank = "-"
    if not _teams_raw.empty and "Team" in _teams_raw.columns:
        _tex_r = _teams_raw[_teams_raw["Team"].astype(str).str.upper() == "TEX"]
        if not _tex_r.empty:
            _r = _tex_r.iloc[0]
            _era_v = pd.to_numeric(pd.Series([_r.get("ERA", float("nan"))]), errors="coerce").iloc[0]
            _fip_v = pd.to_numeric(pd.Series([_r.get("FIP", float("nan"))]), errors="coerce").iloc[0]
            if not pd.isna(_era_v):
                _era_rank = int((pd.to_numeric(_teams_raw["ERA"], errors="coerce") < _era_v).sum() + 1)
                _era_str = f"{_era_v:.2f} (MLB {_era_rank}위)"
            if not pd.isna(_fip_v):
                _fip_rank = int((pd.to_numeric(_teams_raw["FIP"], errors="coerce") < _fip_v).sum() + 1)
                _fip_str = f"{_fip_v:.2f} (MLB {_fip_rank}위)"

    lines += [
        "Team Conclusion",
        f"ERA {_era_str}이면서 -9.06승 잔차 — 이것이 2025 Rangers의 핵심 모순입니다.",
        f"실점 억제는 리그 최고 수준이었으나, 블론 세이브(29회)가 이 이점을 승리로 전환하는 데 실패했습니다.",
        f"FIP {_fip_str}로 ERA-FIP 괴리가 있어 수비·운 보정 시 실질 구위는 ERA보다 낮게 재평가됩니다.",
        "타선 깊이 부재(Seager 이외 중심타선 없음)·불펜 핵심 인원 부상 이탈·9월 급락이 잔차를 구조적으로 누적시켰습니다.",
        "최종 판단은 Simulation 의사결정 후보 순위표를 중심으로 승수 개선 폭, 예측 안정성, 선수 운용 지표를 함께 종합해야 합니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, "TEX 2025 Team Summary Report", lines)
    return buffer.getvalue()


@st.cache_data
def load_data():
    base = DATA_DIR

    return {
        'pitcher_ag': pd.read_csv(base / "pitcher_stats_ag.csv", encoding="utf-8-sig"),
        'pitcher_mb': pd.read_csv(base / "pitcher_stats_mb.csv", encoding="utf-8-sig"),
        'model_comp': pd.read_csv(base / "model_comparison.csv", encoding="utf-8-sig"),
        'model_sum':  pd.read_csv(base / "model_summary.csv", encoding="utf-8-sig"),
        'meta': json.load(open(base / "meta.json", encoding="utf-8")),
    }


data = load_data()
