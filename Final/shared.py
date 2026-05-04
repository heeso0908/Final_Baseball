"""Shared utilities, constants, and data for TEX 2025 dashboard."""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages
import base64
from textwrap import wrap as textwrap_wrap

from simulator import build_scenario_snapshots, run_simulation

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
    "Rotation Spike",
    "Langford Leap",
    "Hopeful Composite",
    "Risk Case",
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
@st.cache_data(show_spinner=False)
def get_simulation_result(raw_dir: str, scenario_name: str, n_sims: int) -> dict:
    return run_simulation(raw_dir, scenario_name, n_sims=n_sims)


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


# ── PDF helpers ───────────────────────────────────────────────
def _add_pdf_page(pdf: PdfPages, title: str, lines: list):
    def _new_page():
        f = plt.figure(figsize=(8.27, 11.69))
        f.patch.set_facecolor("white")
        a = f.add_axes([0, 0, 1, 1])
        a.set_axis_off()
        return f, a

    fig, ax = _new_page()
    ax.text(0.08, 0.95, title, fontsize=16, fontweight="bold", color=TEX_NAVY,
            transform=ax.transAxes, va="top")
    y = 0.88
    for line in lines:
        if line == "":
            y -= 0.020
            continue
        wrapped = textwrap_wrap(line, width=72) or [""]
        for wline in wrapped:
            ax.text(0.08, y, wline, fontsize=10, color="#1B2435",
                    transform=ax.transAxes, va="top")
            y -= 0.030
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig, ax = _new_page()
                y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


@st.cache_data(show_spinner=False)
def build_player_report_pdf(player: str) -> bytes:
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
        "Purpose: 2025 Texas Rangers의 실제 승수와 Pythagorean 기대 승수 간 -9.06승 잔차를 설명하기 위한 선수별 보조 보고서입니다.",
        "",
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
        "이 보고서는 모션 분석을 잔차 분석의 하위 근거로 사용합니다. 단일 선수의 폼만으로 -9.06승 전체를 설명하지 않고, 경기력/선수 운영/시뮬레이션 결과와 함께 해석해야 합니다.",
    ]
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, f"TEX 2025 Player Report - {player}", lines)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def build_team_report_pdf() -> bytes:
    lines = [
        "Purpose",
        "2025 Texas Rangers는 실제 81승, Pythagorean 기대 승수 90.06승으로 -9.06승 잔차를 기록했습니다.",
        "본 대시보드는 이 차이를 설명하기 위해 경기력 분석, 선수 분석, 대표 투수진 모션 분석, 가상 시나리오 시뮬레이션을 순서대로 연결합니다.",
        "",
        "Analysis Flow",
        "1. 경기력 분석: 득실과 실제 승패가 어긋난 구간, 1점차/연장/세이브 상황 등 잔차가 커진 경기 맥락을 확인합니다.",
        "2. 선수 분석: 부상, 타격/투수 지표, 선수별 projection 변화가 팀 승수에 미친 조건을 분리합니다.",
        "3. 대표 투수 모션 분석: 잔차 원인 중 투수 운영과 연결되는 케이스를 3D 키네마틱으로 점검합니다.",
        "4. 시뮬레이션: bullpen, rotation, hitter, composite 시나리오에서 승수 분포와 residual layer가 어떻게 바뀌는지 확인합니다.",
        "5. AI Agent: 업로드된 CSV 기반으로 사용자의 텍스트 질의와 그래프형 질의를 보조합니다.",
        "",
        "Pitcher-Level Summary",
    ]
    for player, finding in REPORT_FINDINGS.items():
        lines.append(f"- {player}: {finding['summary']} Recommendation: {finding['recommendation']}")
    lines.extend([
        "",
        "Team Conclusion",
        "대표 투수 모션 분석은 잔차 -9.06승의 전체 원인이 아니라, 선수/운영 분석을 더 구체화하는 하위 레이어입니다.",
        "Webb처럼 코칭 가능한 폼 분기가 있는 선수와 Garcia처럼 모션 외 요인을 우선 검토해야 하는 선수를 분리하는 것이 핵심입니다.",
        "최종 판단은 시뮬레이션 결과의 승수 분포, 선수별 projection, AI Agent의 CSV 기반 질의 결과를 함께 종합해야 합니다.",
    ])
    buffer = BytesIO()
    with PdfPages(buffer) as pdf:
        _add_pdf_page(pdf, "TEX 2025 Team Summary Report", lines)
    return buffer.getvalue()


# ── Data loading ──────────────────────────────────────────────
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