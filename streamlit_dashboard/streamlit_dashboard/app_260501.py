"""
TEX 2025 Pythagorean residual analysis dashboard.
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import json
from io import BytesIO
from matplotlib.backends.backend_pdf import PdfPages

# 한국어/마이너스 표기 보정: Nanum Gothic이 없으면 시스템 기본 폰트로 fallback됩니다.
plt.rcParams['font.family'] = ['Nanum Gothic', 'Malgun Gothic', 'AppleGothic', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import plotly.express as px
import plotly.graph_objects as go
import altair as alt
import base64
from streamlit_option_menu import option_menu
from textwrap import dedent

from simulator import build_scenario_snapshots, run_simulation

# ─────────────────────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TEX 2025 잔차 분석",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# TEX 팀 컬러 + Lovable 스타일
TEX_BLUE = "#003278"
TEX_NAVY = "#0B1F3A"
TEX_RED = "#C0111F"
TEX_LIGHT = "#F6F8FB"
TEX_MUTED = "#64748B"

# ─────────────────────────────────────────────────────────────
# 시뮬레이션 설정: app.py + simulator.py 연동
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
RAW_DIR = BASE_DIR / "data_raw"

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

st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");

/* ─────────────────────────────
   Lovable Theme Variables
───────────────────────────── */
:root {
    --navy: #0D1B33;
    --navy-deep: #071225;
    --navy-soft: #243A5E;
    --rangers-red: #B31922;
    --rangers-red-soft: #D04A52;

    --background: #FAFBFC;
    --foreground: #1B2435;
    --card: #FFFFFF;
    --muted: #F3F5F8;
    --muted-foreground: #667085;
    --border: #E4E8EF;
    --grid: #E9EDF3;

    --positive: #2F9E65;
    --negative: #B31922;
    --neutral: #B8BDC7;

    --radius: 14px;
    --shadow-card: 0 1px 2px rgba(13, 27, 51, 0.04), 0 1px 3px rgba(13, 27, 51, 0.06);
    --shadow-elevated: 0 4px 6px -1px rgba(13, 27, 51, 0.06), 0 10px 24px -8px rgba(13, 27, 51, 0.10);
}

/* ─────────────────────────────
   Base
───────────────────────────── */
html, body, [class*="css"] {
    font-family: "Manrope", "Pretendard", "Noto Sans KR", system-ui, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 8% 0%, rgba(13, 27, 51, 0.055), transparent 28%),
        radial-gradient(circle at 92% 8%, rgba(179, 25, 34, 0.045), transparent 24%),
        var(--background);
    color: var(--foreground);
}

.block-container {
    max-width: 1600px;
    padding-top: 3.2rem !important;
    padding-bottom: 3rem;
    padding-left: 3.2rem !important;
    padding-right: 2rem;
}

/* Streamlit 기본 상단바와 본문 간격 보정 */
[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stToolbar"] {
    right: 1rem !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: "Sora", "Manrope", sans-serif;
    letter-spacing: -0.018em;
    color: var(--navy);
}

h1 {
    font-size: 34px !important;
    line-height: 1.16 !important;
    font-weight: 800 !important;
    margin-bottom: 0.45rem !important;
}

h2 {
    font-size: 23px !important;
    line-height: 1.25 !important;
    font-weight: 700 !important;
    margin-top: 1.6rem !important;
}

h3 {
    font-size: 17px !important;
    line-height: 1.35 !important;
    font-weight: 700 !important;
}

p, li, .stMarkdown {
    font-size: 14px;
    line-height: 1.65;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(13, 27, 51, 0.10), transparent);
    margin: 28px 0;
}

.num {
    font-family: "JetBrains Mono", "Sora", monospace;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
}


/* ─────────────────────────────
   Sidebar — Lovable Style
───────────────────────────── */
/*
  구조 핵심
  - 사이드바 자체의 상단 118px을 흰색으로 칠함
  - 로고 박스는 투명 + 100% 폭으로 중앙 정렬만 담당
  - 사이드바 내부 wrapper padding을 제거해 오른쪽 스크롤/여백이 튀지 않게 처리
*/
section[data-testid="stSidebar"] {
    background: linear-gradient(
        to bottom,
        #FFFFFF 0px,
        #FFFFFF 118px,
        #071A35 118px,
        #071A35 100%
    ) !important;
    min-width: 285px !important;
    max-width: 285px !important;
    width: 285px !important;
    border-right: 1px solid rgba(13, 27, 51, 0.10);
}

/* 사이드바 접기 버튼/header 영역 제거 */
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Streamlit 사이드바 내부 기본 padding 제거 */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebarContent"],
section[data-testid="stSidebarUserContent"] {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    overflow-x: hidden !important;
}

/* 로고가 들어간 markdown wrapper가 폭을 좁히는 문제 방지 */
section[data-testid="stSidebar"] .element-container:has(.sidebar-brand-shell),
section[data-testid="stSidebar"] .stMarkdown:has(.sidebar-brand-shell),
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.sidebar-brand-shell) {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* 로고 영역: 사이드바 배경의 흰색 영역을 그대로 사용 */
.sidebar-brand-shell {
    width: 100% !important;
    min-width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    background: transparent !important;
    padding: 16px 0 16px 0 !important;
    min-height: 118px;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    box-sizing: border-box;
    border-bottom: 1px solid rgba(13, 27, 51, 0.10);
}

.sidebar-logo-full {
    width: 188px !important;
    max-width: 82% !important;
    height: auto !important;
    object-fit: contain;
    display: block !important;
    margin: 0 auto !important;
}

/* 프로젝트 정보 영역 */
.sidebar-project-wrap {
    padding: 22px 18px 16px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 12px;
}

.sidebar-section-label {
    color: rgba(255,255,255,0.58);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.sidebar-project-title {
    color: #FFFFFF;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 17px;
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
}

.sidebar-project-sub {
    color: rgba(255,255,255,0.70);
    font-size: 14px;
    font-weight: 500;
    margin-top: 5px;
}

.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 12px 18px 14px 18px;
}

.sidebar-note-wrap {
    padding: 6px 18px 0 18px;
    text-align: left;
    box-sizing: border-box;
}

.sidebar-note-body {
    color: rgba(255,255,255,0.68);
    font-size: 12px;
    line-height: 1.5;
    text-align: left;
}

/* ─────────────────────────────
   Hero
───────────────────────────── */
.hero-card {
    position: relative;
    overflow: hidden;
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(179,25,34,0.22) 0%, transparent 52%),
        linear-gradient(135deg, #0D1B33 0%, #142A4F 58%, #0A1529 100%);
    border-radius: 24px;
    padding: 28px 30px;
    box-shadow: 0 18px 42px -18px rgba(13, 27, 51, 0.55);
    margin-top: 0.5rem;
    margin-bottom: 22px;
    border: 1px solid rgba(255,255,255,0.10);
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -60px;
    bottom: -70px;
    border-radius: 999px;
    background: rgba(255,255,255,0.06);
}

.hero-card h1 {
    color: white !important;
    font-size: 33px !important;
    line-height: 1.12 !important;
    font-weight: 800 !important;
    margin: 9px 0 8px 0 !important;
    position: relative;
    z-index: 1;
}

.hero-card p {
    color: rgba(255,255,255,0.73);
    font-size: 14px;
    line-height: 1.65;
    max-width: none;
    width: 100%;
    margin: 0;
    position: relative;
    z-index: 1;
}

/* ─────────────────────────────
   Pills
───────────────────────────── */
.pill {
    display: inline-flex;
    align-items: center;
    padding: 5px 10px;
    border-radius: 999px;
    font-family: "Manrope", sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.01em;
    border: 1px solid rgba(13, 27, 51, 0.08);
    background: white;
    color: #344054;
    margin-right: 7px;
    margin-bottom: 6px;
    position: relative;
    z-index: 1;
}

.pill-red {
    background: rgba(179, 25, 34, 0.10);
    color: var(--rangers-red);
    border-color: rgba(179, 25, 34, 0.18);
}

.pill-navy {
    background: rgba(13, 27, 51, 0.10);
    color: var(--navy);
    border-color: rgba(13, 27, 51, 0.16);
}

.pill-white {
    background: rgba(255,255,255,0.13);
    color: white;
    border-color: rgba(255,255,255,0.20);
}

/* ─────────────────────────────
   Glass / Cards
───────────────────────────── */
.card,
.glass-card,
.kpi-card-custom {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.78) 100%);
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.72);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 8px 24px -12px rgba(13,27,51,0.22),
        0 2px 6px -2px rgba(13,27,51,0.10);
}

.card {
    border-radius: 18px;
    padding: 26px 32px !important;
    margin-bottom: 16px;
}

.card:hover,
.glass-card:hover,
.kpi-card-custom:hover {
    transform: translateY(-1px);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 14px 32px -14px rgba(13,27,51,0.28),
        0 4px 10px -3px rgba(13,27,51,0.12);
    transition: all 0.18s ease;
}

.glass-card {
    border-radius: 20px;
    padding: 22px;
}

.glass-card-accent {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(179,25,34,0.08) 0%, transparent 55%),
        linear-gradient(145deg, rgba(255,255,255,0.92), rgba(255,255,255,0.74));
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid rgba(179,25,34,0.14);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 10px 28px -14px rgba(179,25,34,0.30),
        0 2px 6px -2px rgba(13,27,51,0.10);
}

/* KPI */
.kpi-card-custom {
    position: relative;
    border-radius: 18px;
    padding: 18px 18px 16px 20px;
    min-height: 112px;
    overflow: hidden;
    border: 1px solid rgba(13, 27, 51, 0.07);
    box-shadow:
        0 14px 32px -18px rgba(13, 27, 51, 0.30),
        0 3px 8px -4px rgba(13, 27, 51, 0.14);
    transition: all 0.18s ease;
}

.kpi-card-custom:hover {
    transform: translateY(-2px);
    box-shadow:
        0 18px 38px -18px rgba(13, 27, 51, 0.36),
        0 5px 12px -4px rgba(13, 27, 51, 0.16);
}

.kpi-card-custom::before {
    content: "";
    position: absolute;
    left: 0;
    top: 14px;
    bottom: 14px;
    width: 5px;
    border-radius: 999px;
}

.kpi-card-custom.accent-navy {
    background:
        radial-gradient(120% 100% at 100% 0%, rgba(13, 27, 51, 0.075) 0%, transparent 54%),
        linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
}

.kpi-card-custom.accent-navy::before {
    background: linear-gradient(180deg, #17325C 0%, #0D1B33 100%);
}

.kpi-card-custom.accent-red {
    background:
        radial-gradient(120% 100% at 100% 0%, rgba(179, 25, 34, 0.10) 0%, transparent 54%),
        linear-gradient(180deg, #ffffff 0%, #fff7f8 100%);
}

.kpi-card-custom.accent-red::before {
    background: linear-gradient(180deg, #E13D45 0%, #B31922 100%);
}

.kpi-card-custom.accent-gray {
    background:
        radial-gradient(120% 100% at 100% 0%, rgba(120, 130, 150, 0.06) 0%, transparent 54%),
        linear-gradient(180deg, #ffffff 0%, #fafbfc 100%);
}

.kpi-card-custom.accent-gray::before {
    background: linear-gradient(180deg, #C5CCD6 0%, #8F9AAA 100%);
}

.kpi-label-custom {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5F6B7A;
    margin-bottom: 11px;
}

.kpi-value-custom {
    font-family: "Sora", "JetBrains Mono", sans-serif;
    font-size: 26px;
    line-height: 1.08;
    font-weight: 700;
    letter-spacing: -0.035em;
    color: #0D1B33;
    margin-bottom: 7px;
}

.kpi-sub-custom {
    font-size: 12px;
    color: #7A8495;
    line-height: 1.45;
}

/* finding */
.finding-box {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(179,25,34,0.08) 0%, transparent 55%),
        linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,255,255,0.72));
    border: 1px solid rgba(179,25,34,0.14);
    color: #344054;
    border-radius: 18px;
    padding: 16px 18px;
    font-size: 13px;
    line-height: 1.65;
    margin: 15px 0;
    box-shadow: 0 10px 28px -18px rgba(179,25,34,0.28);
}

.finding-box strong {
    color: var(--rangers-red);
}

/* Streamlit metric도 남아 있는 곳 대비 */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(13, 27, 51, 0.08);
    border-radius: 18px;
    padding: 16px;
    box-shadow: var(--shadow-card);
}

[data-testid="stMetricValue"] {
    color: var(--navy);
    font-family: "Sora", "JetBrains Mono", sans-serif;
    font-size: 24px !important;
    font-weight: 800;
}

[data-testid="stMetricLabel"] {
    color: var(--muted-foreground);
    font-size: 11px !important;
    font-weight: 700;
}

/* ─────────────────────────────
   Tables / Tabs / Forms
───────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(13, 27, 51, 0.08);
    box-shadow: var(--shadow-card);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 999px;
    padding: 8px 14px;
    background: rgba(255,255,255,0.72);
    border: 1px solid rgba(13,27,51,0.07);
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background: var(--navy) !important;
    color: white !important;
}

[data-testid="stAlert"] {
    border-radius: 16px;
    border: 1px solid rgba(13,27,51,0.08);
    box-shadow: var(--shadow-card);
}

.stButton > button {
    border-radius: 12px;
    border: 1px solid rgba(13, 27, 51, 0.08);
    font-weight: 700;
}

div[data-baseweb="select"] > div {
    border-radius: 12px;
}

/* matplotlib figure background 보정 */
[data-testid="stImage"],
[data-testid="stVideo"] {
    border-radius: 18px;
    overflow: hidden;
}

.question-box {
    background: rgba(13, 27, 51, 0.045);
    border: 1px solid rgba(13, 27, 51, 0.08);
    border-left: 4px solid #0D1B33;
    border-radius: 16px;
    padding: 16px 18px;
    color: #1B2435;
    font-size: 14px;
    line-height: 1.7;
    box-shadow: 0 10px 24px -18px rgba(13, 27, 51, 0.24);
    margin-top: 18px;
    max-width: 900px;
}

.question-box strong {
    color: #0D1B33;
}

.target-panel {
    padding-left: 34px;
    border-left: 1px solid rgba(13, 27, 51, 0.12);
    margin-left: 16px;
}

.target-panel h3 {
    margin-top: 0 !important;
    margin-bottom: 26px !important;
    font-size: 17px !important;
    color: #1B2435;
}

.target-list {
    display: flex;
    flex-direction: column;
    gap: 22px;
}

.target-item {
    margin: 0;
}

.target-name {
    font-size: 14px;
    font-weight: 800;
    color: #1B2435;
    margin-bottom: 6px;
}

.target-name span {
    font-weight: 500;
    color: #667085;
}

.target-desc {
    font-size: 13px;
    color: #344054;
}


/* ─────────────────────────────
   Page-level Lovable Components
───────────────────────────── */
.page-kicker {
    color: rgba(255,255,255,0.70);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    position: relative;
    z-index: 1;
}
.section-shell {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(179,25,34,0.055) 0%, transparent 54%),
        linear-gradient(145deg, rgba(255,255,255,0.94) 0%, rgba(255,255,255,0.78) 100%);
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid rgba(255,255,255,0.74);
    border-radius: 20px;
    padding: 20px 22px;
    margin: 14px 0 18px 0;
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.92) inset,
        0 10px 28px -16px rgba(13,27,51,0.22),
        0 2px 8px -4px rgba(13,27,51,0.12);
}
.section-shell.compact {
    padding: 16px 18px;
    margin: 10px 0 14px 0;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.78) 100%) !important;
    backdrop-filter: blur(14px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(140%) !important;
    border: 1px solid rgba(255,255,255,0.72) !important;
    border-radius: 20px !important;
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 8px 24px -12px rgba(13,27,51,0.22),
        0 2px 6px -2px rgba(13,27,51,0.10) !important;
}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.72rem;
}
.section-heading {
    font-family: "Sora", "Manrope", sans-serif;
    color: #0D1B33;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}
.section-copy {
    color: #667085;
    font-size: 13px;
    line-height: 1.65;
    margin: 0 0 12px 0;
}
.chart-shell {
    background:
        linear-gradient(180deg, rgba(255,255,255,0.96), rgba(249,251,255,0.82));
    border: 1px solid rgba(13, 27, 51, 0.07);
    border-radius: 18px;
    padding: 16px 16px 10px 16px;
    margin-bottom: 14px;
    box-shadow: 0 10px 28px -18px rgba(13, 27, 51, 0.22);
}
.chart-title {
    font-family: "Sora", "Manrope", sans-serif;
    color: #0D1B33;
    font-size: 16px;
    font-weight: 800;
    letter-spacing: -0.015em;
    margin-bottom: 4px;
}
.chart-caption {
    color: #667085;
    font-size: 12px;
    line-height: 1.55;
    margin-bottom: 12px;
}
.inline-stat-grid {
    display: grid;
    grid-template-columns: repeat(5, minmax(0, 1fr));
    gap: 10px;
}
.pitcher-mini-card {
    background: rgba(255,255,255,0.76);
    border: 1px solid rgba(13,27,51,0.08);
    border-radius: 16px;
    padding: 14px;
    min-height: 122px;
    box-shadow: 0 8px 20px -16px rgba(13,27,51,0.24);
}
.pitcher-mini-name {
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 15px;
    font-weight: 800;
    color: #0D1B33;
    margin-bottom: 4px;
}
.pitcher-mini-role {
    color: #667085;
    font-size: 12px;
    font-weight: 700;
    margin-bottom: 8px;
}
.pitcher-mini-tag {
    display: inline-flex;
    padding: 4px 8px;
    border-radius: 999px;
    background: rgba(179,25,34,0.09);
    color: #B31922;
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 8px;
}
.pitcher-mini-note {
    color: #344054;
    font-size: 12px;
    line-height: 1.5;
}
.quote-card {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(13,27,51,0.08), transparent 58%),
        linear-gradient(135deg, rgba(13,27,51,0.96), rgba(20,42,79,0.94));
    color: white;
    border-radius: 20px;
    padding: 22px 24px;
    box-shadow: 0 18px 42px -20px rgba(13,27,51,0.52);
}
.quote-card p, .quote-card li, .quote-card strong {
    color: white !important;
}
@media (max-width: 1100px) {
    .inline-stat-grid { grid-template-columns: 1fr 1fr; }
}

/* Streamlit border container를 Lovable 카드처럼 보이게 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background:
        linear-gradient(145deg, rgba(255,255,255,0.92) 0%, rgba(255,255,255,0.78) 100%) !important;
    backdrop-filter: blur(14px) saturate(140%) !important;
    -webkit-backdrop-filter: blur(14px) saturate(140%) !important;
    border: 1px solid rgba(255,255,255,0.72) !important;
    border-radius: 20px;
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 8px 24px -12px rgba(13,27,51,0.22),
        0 2px 6px -2px rgba(13,27,51,0.10) !important;
    padding: 20px 22px !important;
}

.section-heading {
    font-family: "Sora", "Manrope", sans-serif;
    color: #0D1B33;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 0 0 6px 0;
}

.section-copy {
    color: #667085;
    font-size: 13px;
    line-height: 1.65;
    margin: 0 0 12px 0;
}

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Lovable 스타일 Helper Components
# ─────────────────────────────────────────────────────────────
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
    st.markdown(f"""
    <div class="glass-card">
        {body}
    </div>
    """, unsafe_allow_html=True)
    



def page_hero(kicker: str, title: str, body: str, badges: list[tuple[str, str]] | None = None):
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


# ─────────────────────────────────────────────────────────────
# 시뮬레이션 Lazy Loader
# - 앱 첫 로딩 속도를 위해 버튼 클릭 시에만 simulator.py 실행
# - series_summary는 최종 화면에서 제외
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_simulation_result(raw_dir: str, scenario_name: str, n_sims: int) -> dict:
    return run_simulation(raw_dir, scenario_name, n_sims=n_sims)


@st.cache_data(show_spinner=False)
def get_scenario_snapshots(raw_dir: str) -> dict:
    return build_scenario_snapshots(raw_dir)


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


REPORT_FINDINGS = {
    "Leiter": {
        "role": "Starter",
        "summary": "SO vs Walk에서 통계적으로 강한 모션 분기는 제한적입니다.",
        "recommendation": "폼 교정보다는 피칭 디자인, 구종 선택, 카운트별 운영을 우선 점검합니다.",
    },
    "Webb": {
        "role": "Starter",
        "summary": "HSS와 Trunk/Hip ratio에서 뚜렷한 차이가 확인된 핵심 코칭 대상입니다.",
        "recommendation": "하체-상체 분리와 릴리스 전 회전 타이밍을 안정화하는 루틴을 설계합니다.",
    },
    "Garcia": {
        "role": "Closer",
        "summary": "SV vs BS 간 모션 차이가 작아 null finding으로 해석하는 편이 적절합니다.",
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


def _add_pdf_page(pdf: PdfPages, title: str, lines: list[str]):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    fig.text(0.08, 0.94, title, fontsize=18, fontweight="bold", color=TEX_NAVY)
    y = 0.88
    for line in lines:
        if line == "":
            y -= 0.026
            continue
        fig.text(0.08, y, line, fontsize=10.5, color="#1B2435", va="top", wrap=True)
        y -= 0.034
        if y < 0.08:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.patch.set_facecolor("white")
            y = 0.94
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


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


# ─────────────────────────────────────────────────────────────
# 데이터 로딩 (캐시)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base = Path(__file__).parent / "data"
    return {
        'pitcher_ag': pd.read_csv(base / "pitcher_stats_ag.csv", encoding="utf-8-sig"),
        'pitcher_mb': pd.read_csv(base / "pitcher_stats_mb.csv", encoding="utf-8-sig"),
        'model_comp': pd.read_csv(base / "model_comparison.csv", encoding="utf-8-sig"),
        'model_sum': pd.read_csv(base / "model_summary.csv", encoding="utf-8-sig"),
        'meta': json.load(open(base / "meta.json", encoding="utf-8")),
    }

data = load_data()
ASSETS = Path(__file__).parent / "assets"


# ─────────────────────────────────────────────────────────────
# 사이드바 네비게이션 - Lovable 스타일
# ─────────────────────────────────────────────────────────────
def image_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


logo_path = ASSETS / "images" / "logo.png"

with st.sidebar:
    # 브랜드 로고 영역
    if logo_path.exists():
        logo_base64 = image_to_base64(logo_path)
        st.markdown(f"""
        <div class="sidebar-brand-shell">
            <img class="sidebar-logo-full" src="data:image/png;base64,{logo_base64}">
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-brand-shell">
            <div style="font-family:Sora, Manrope, sans-serif; color:#0D1B33; font-size:18px; font-weight:800; line-height:1.05;">
                Monday<br>Likes Baseball
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 프로젝트 정보 영역
    st.markdown("""
    <div class="sidebar-project-wrap">
        <div class="sidebar-section-label">Client Project</div>
        <div class="sidebar-project-title">Texas Rangers</div>
        <div class="sidebar-project-sub">2025 Residual Analysis</div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    # 메뉴
    selected = option_menu(
        menu_title=None,
        options=[
            "Overview",
            "Simulation",
            "Methodology",
            "Leiter",
            "Webb",
            "Garcia",
            "Armstrong",
            "Jackson",
            "Comparison",
            "AI Agent",
            "Conclusions",
        ],
        icons=[
            "grid",
            "activity",
            "bezier2",
            "person-fill",
            "person",
            "person-fill",
            "person",
            "person-fill",
            "bar-chart",
            "chat-dots",
            "pin-angle",
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0 8px!important",
                "margin": "0!important",
                "background-color": "#071A35",
            },
            "icon": {
                "color": "rgba(255,255,255,0.80)",
                "font-size": "16px",
            },
            "nav-link": {
                "font-family": "Manrope, sans-serif",
                "font-size": "14px",
                "font-weight": "600",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "11px 14px",
                "color": "rgba(255,255,255,0.88)",
                "border-radius": "10px",
                "background-color": "#071A35",
                "border-left": "3px solid transparent",
            },
            "nav-link:hover": {
                "background-color": "#10264A",
                "color": "white",
            },
            "nav-link-selected": {
                "background-color": "#1A3257",
                "color": "white",
                "font-weight": "700",
                "border-left": "3px solid #B31922",
            },
        }
    )

    st.markdown("""
    <div class="sidebar-divider"></div>
    <div class="sidebar-note-wrap">
        <div class="sidebar-section-label">Project Note</div>
        <div class="sidebar-note-body">
            TEX 2025 잔차 -9.06승 원인 진단<br>
            — 경기력 · 선수 · 모션 · 시뮬레이션
        </div>
    </div>
    """, unsafe_allow_html=True)


# 선택 메뉴를 기존 page 라우팅 값으로 변환
PAGE_MAP = {
    "Overview": "overview",
    "Simulation": "simulation",
    "Methodology": "methodology",
    "Leiter": "leiter",
    "Webb": "webb",
    "Garcia": "garcia",
    "Armstrong": "armstrong",
    "Jackson": "jackson",
    "Comparison": "comparison",
    "AI Agent": "ai_agent",
    "Conclusions": "conclusions",
}

page = PAGE_MAP[selected]

# ─────────────────────────────────────────────────────────────
# Page: Overview
# ─────────────────────────────────────────────────────────────
def show_overview():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">Pythagorean Residual</span>
        <span class="pill pill-white">Game & Player Analysis</span>
        <span class="pill pill-white">Scenario Simulation</span>
        <h1>2025 TEX 잔차 분석 대시보드</h1>
        <p>
        이 대시보드의 목표는 2025 텍사스 레인저스가 실제 승수보다 Pythagorean 기대 승수에서 9.06승 낮게 끝난 이유를 설명하는 것입니다.
        경기력과 선수 맥락을 먼저 분해하고, 그 하위 근거로 대표 투수진의 3D 모션 분석과 가상 시나리오 시뮬레이션을 연결합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # 핵심 KPI 카드
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        kpi_card("잔차 규모", "-9.06승", "Actual 81W vs Pythag 90.06W", accent="red")

    with col2:
        kpi_card("분석 흐름", "5단계", "경기력 → 선수 → 모션 → 시뮬 → 보고서", accent="navy")

    with col3:
        kpi_card("대표 투수", "5명", "잔차 원인 세부 케이스", accent="red")

    with col4:
        kpi_card("AI Agent", "CSV 기반", "텍스트·그래프 질의 지원", accent="navy")
    
    st.markdown("---")
    
    # 프로젝트 배경
    st.markdown("## 분석 배경")

    left_col, right_col = st.columns([1.55, 1], gap="large")

    with left_col:
        st.markdown("""
        **2025 텍사스 레인저스의 Pythagorean 잔차 -9.06승**  
        팀이 기대 승수보다 9승가량 적게 기록한 격차가 출발점입니다.
        따라서 본 프로젝트의 중심 질문은 "투수 폼이 달랐는가"가 아니라
        **왜 실제 승수와 기대 승수 사이에 큰 잔차가 발생했는가**입니다.

        이를 위해 경기력 분석과 선수 분석으로 팀 단위 맥락을 먼저 확인한 뒤,
        잔차와 연결될 수 있는 투수 운영 케이스를 더 깊게 보기 위해 5명 대표 투수의
        결과 분기 상황(SO/Walk, SV/BS)을 3D 키네마틱 지표로 검증합니다.\
        """)

        st.markdown("""
        <div class="question-box">
            <strong>핵심 질문:</strong><br>
            81승 팀이 왜 90.06승 기대치에 미치지 못했는가?
            그 차이는 경기 운영, 선수 상태, 투수 모션, 가상 시나리오에서 어떻게 설명되는가?
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        target_items = []
        for pitcher, info in data["meta"]["pitchers"].items():
            target_items.append(
                f'<div class="target-item">'
                f'<div class="target-name">{pitcher} <span>({info["role"]})</span></div>'
                f'<div class="target-desc">{info["situation"]} (n={info["n_a"]}/{info["n_b"]})</div>'
                f'</div>'
            )

        st.markdown(
            f'<div class="target-panel">'
            f'<h3>대표 투수 모션 케이스</h3>'
            f'<div class="target-list">{"".join(target_items)}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    
    # 분석 흐름
    st.markdown("## 분석 흐름")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="card">
            <span class="pill pill-red">1. Residual Diagnosis</span>
            <h3>경기력 분석</h3>
            <p><b>득실과 승패가 어긋난 구간 확인</b></p>
            <ul>
                <li>1점차, 연장, 세이브 상황</li>
                <li>월별 득실·스케줄 맥락</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ -9.06승 잔차의 발생 지점 탐색</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="card">
            <span class="pill pill-navy">2. Player Layer</span>
            <h3>선수 분석</h3>
            <p><b>타자·투수 projection 변화 분해</b></p>
            <ul>
                <li>부상과 역할 변화</li>
                <li>타자/투수별 시나리오 카드</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 팀 잔차를 선수 단위 원인으로 좁힘</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="card">
            <span class="pill">3. Motion & Simulation</span>
            <h3>모션 분석 + 가상 시뮬레이션</h3>
            <p><b>대표 투수 케이스를 승수 시나리오와 연결</b></p>
            <ul>
                <li>Webb·Garcia 등 선수별 원인 분리</li>
                <li>조건 변화별 승수 분포 확인</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 어떤 조합에서 잔차가 완화되는지 검증</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 사용 안내
    st.markdown("## 대시보드 사용법")
    st.markdown("""
    좌측 사이드바에서 페이지 이동:
    
    - **Simulation**: 2025 시즌 재구성 및 시나리오별 승수 분포
    - **Methodology**: 사용한 Pose 모델 비교
    - **각 투수**: 잔차 원인 세부화를 위한 대표 투수 모션 분석
    - **Comparison**: 5명 통합 비교
    - **AI Agent**: 업로드된 CSV 기반 텍스트/그래프 질의
    - **Conclusions**: 종합 결론 및 PDF 보고서 출력
    """)  


# ─────────────────────────────────────────────────────────────
# Page: Methodology
# ─────────────────────────────────────────────────────────────
def show_methodology():
    page_hero(
        "Methodology",
        "Pose Estimation Pipeline",
        "잔차 분석의 하위 레이어인 대표 투수 모션 분석을 위해 MotionBERT와 MotionAGFormer의 측정 안정성을 비교하고, 더 안정적인 MotionAGFormer 기반 키네마틱 지표를 채택한 과정입니다.",
        [("MotionBERT", "white"), ("MotionAGFormer", "white"), ("CV Stability", "white")],
    )

    df_model = data['model_sum'].pivot_table(
        index='metric', columns='model', values='cv_pct'
    ).reset_index()

    top_left, top_right = st.columns([1.1, 1], gap="large")

    with top_left:
        with st.container(border=True):
            st.markdown(
                '<div class="section-heading">모델별 측정 안정성</div>',
                unsafe_allow_html=True
            )
            st.markdown(
                '<div class="section-copy">Coefficient of Variation(CV%) 기준으로 낮을수록 반복 측정 안정성이 높습니다.</div>',
                unsafe_allow_html=True
            )

            if 'MotionBERT' in df_model.columns and 'MotionAGFormer' in df_model.columns:
                df_model['개선'] = df_model['MotionBERT'] - df_model['MotionAGFormer']
                df_model = df_model.round(2)

                st.dataframe(df_model, use_container_width=True, hide_index=True)

                avg_improvement = df_model['개선'].mean()
                if avg_improvement > 0:
                    finding_box(
                        "모델 선택 근거",
                        f"평균 CV가 <strong>{avg_improvement:.1f}%p</strong> 감소해 MotionAGFormer를 최종 채택했습니다."
                    )
            else:
                st.info("모델 비교 컬럼을 확인해주세요.")

    with top_right:
        with st.container(border=True):
            st.markdown('<div class="section-heading">측정 지표 정의</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">투구 폼의 분리, 회전, 타이밍을 설명하는 핵심 3D 키네마틱 지표입니다.</div>', unsafe_allow_html=True)
            metrics_info = pd.DataFrame({
                '지표': ['HSS at FP', 'Hip peak 3D', 'Trunk peak 3D', 'Trunk/Hip ratio', 'Timing diff', 'HSS max'],
                '의미': [
                    'Foot Plant 시점의 Hip-Shoulder Separation',
                    '골반 회전 최대 각속도 (3D)',
                    '몸통 회전 최대 각속도 (3D)',
                    '몸통/골반 회전 비율 (kinetic chain)',
                    '몸통 vs 골반 피크 타이밍 차이',
                    'Hip-Shoulder Separation 최대값',
                ],
                '단위': ['°', '°/s', '°/s', '비율', 'ms', '°'],
            })
            st.dataframe(metrics_info, use_container_width=True, hide_index=True)

    section_header("시각적 품질 비교", "Baseline부터 smoothing 제거, bone length normalization까지 모델 개선 과정을 이미지로 확인합니다.")
    img_dir = ASSETS / "images"
    tabs = st.tabs(["Baseline", "v2a Smooth 제거", "v2b Bone fix", "전체 비교"])
    image_specs = [
        ("baseline_quality.png", "초기 baseline 품질"),
        ("v2a_nosmooth_quality.png", "Smoothing 제거 후 raw signal 보존"),
        ("v2b_bonefix_quality.png", "Bone length 정규화 적용"),
        ("version_comparison.png", "전체 버전 비교"),
    ]
    for tab, (filename, caption) in zip(tabs, image_specs):
        with tab:
            path = img_dir / filename
            if path.exists():
                st.image(str(path), use_container_width=True)
                st.caption(caption)
            else:
                st.info(f"이미지 파일 없음: {filename}")



# ─────────────────────────────────────────────────────────────
# 범용 투수 페이지 함수
# ─────────────────────────────────────────────────────────────
def show_pitcher_page(pitcher_name, situation_a, situation_b, 
                       interpretation_md, key_findings):
    """
    범용 투수 페이지
    """
    role_kr = "선발" if pitcher_name in ["Leiter", "Webb"] else (
        "마무리" if pitcher_name == "Garcia" else "불펜"
    )
    page_hero(
        "Pitcher Detail",
        f"{pitcher_name} · {role_kr}",
        f"잔차 분석을 선수 단위로 좁히기 위해 {situation_a}와 {situation_b} 상황에서 3D pose 기반 키네마틱 지표와 영상 차이를 비교합니다.",
        [(f"{situation_a} vs {situation_b}", "white"), (role_kr, "white"), ("MotionAGFormer", "white")],
    )

    df = data['pitcher_ag']
    pitcher_df = df[df['player'] == pitcher_name]
    n_a = pitcher_df['n_a'].iloc[0] if len(pitcher_df) > 0 else 0
    n_b = pitcher_df['n_b'].iloc[0] if len(pitcher_df) > 0 else 0

    top_line_left, top_line_right = st.columns([1.4, 1], gap="large")
    with top_line_left:
        with st.container(border=True):
            st.markdown('<div class="section-heading">영상 비교</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">원본과 스켈레톤 오버레이를 전환하며 같은 투수의 성공/실패 장면을 비교합니다.</div>', unsafe_allow_html=True)
            mode = st.radio(
                "영상 모드",
                ["원본", "스켈레톤"],
                horizontal=True,
                key=f"{pitcher_name}_mode"
            )
            folder = "skeleton" if mode == "스켈레톤" else "original"
            case_a = f"{pitcher_name.lower()}_{situation_a.lower()}"
            case_b = f"{pitcher_name.lower()}_{situation_b.lower()}"
            video_a = ASSETS / "videos" / folder / f"{case_a}.mp4"
            video_b = ASSETS / "videos" / folder / f"{case_b}.mp4"
            video_col_1, video_col_2 = st.columns(2, gap="medium")
            with video_col_1:
                st.markdown(f"#### ✅ {situation_a} <span style='color:#667085; font-size:13px;'>(n={n_a})</span>", unsafe_allow_html=True)
                if video_a.exists():
                    st.video(str(video_a))
                else:
                    st.warning(f"영상 없음: {case_a}.mp4")
            with video_col_2:
                st.markdown(f"#### ❌ {situation_b} <span style='color:#667085; font-size:13px;'>(n={n_b})</span>", unsafe_allow_html=True)
                if video_b.exists():
                    st.video(str(video_b))
                else:
                    st.warning(f"영상 없음: {case_b}.mp4")

    with top_line_right:
        with st.container(border=True):
            st.markdown('<div class="section-heading">핵심 키네마틱 지표</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">Cohen&#39;s d 절대값이 큰 순서로 상위 지표를 요약했습니다.</div>', unsafe_allow_html=True)
            for label, mean_a, mean_b, p_val, d in key_findings:
                delta_str = f"{mean_a - mean_b:+.2f}"
                sig_text = "유의" if p_val < 0.05 else "n.s."
                accent = "red" if p_val < 0.05 else "navy"
                kpi_card(label, f"{mean_a:.2f}", f"Δ {delta_str} | p={p_val:.3f} | d={d:+.2f} ({sig_text})", accent=accent)

    stat_left, stat_right = st.columns([1.2, 1], gap="large")
    with stat_left:
        with st.container(border=True):
            st.markdown('<div class="section-heading">통계 검정 결과</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-copy">p-value가 낮고 효과 크기 |d|가 클수록 상황별 폼 차이가 뚜렷합니다.</div>', unsafe_allow_html=True)
            if len(pitcher_df) > 0:
                display_df = pitcher_df[['label', 'a_mean', 'b_mean', 'diff', 'cohens_d', 'u_p', 't_p']].copy()
                display_df.columns = ['지표', f'{situation_a} 평균', f'{situation_b} 평균', '차이', "Cohen's d", 'p (Mann-Whitney)', 'p (t-test)']
                for col in display_df.columns[1:]:
                    display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(3)
                def highlight_sig(row):
                    try:
                        p = float(row['p (Mann-Whitney)'])
                        if p < 0.01:
                            return ['background-color: #FFE4E1'] * len(row)
                        elif p < 0.05:
                            return ['background-color: #FFF8DC'] * len(row)
                    except Exception:
                        pass
                    return [''] * len(row)
                st.dataframe(display_df.style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)
                st.caption("🟥 p<0.01 | 🟨 p<0.05")
            else:
                st.info("해당 투수의 통계 데이터가 없습니다.")

    with stat_right:
        with st.container(border=True):
            st.markdown('<div class="section-heading">분석 해석</div>', unsafe_allow_html=True)
            st.markdown(interpretation_md)



# ─────────────────────────────────────────────────────────────
# 5명 투수 페이지 — 각 선수별 호출
# ─────────────────────────────────────────────────────────────
def show_leiter():
    df = data['pitcher_ag']
    leiter_df = df[df['player'] == 'Leiter'].copy()
    
    # 핵심 지표 4개 (Cohen's d 큰 순)
    top4 = leiter_df.reindex(leiter_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    
    interpretation = """
**선발 투수 Leiter — SO vs Walk 분기**

분석 결과 통계적으로 강한 차이는 발견되지 않음 (대부분 p > 0.1).
이는 다음을 시사:

- **폼 자체는 일관**되게 유지됨
- 결과 차이는 폼이 아닌 **다른 요인**에서 비롯될 가능성
  - 구속/구질 변화
  - 제구점 (location)
  - 타자 매치업
  - 카운트 운영

**시사점**: 폼 교정보다 **피칭 디자인** 측면 검토 필요.
"""
    
    show_pitcher_page("Leiter", "SO", "Walk", interpretation, key_findings)


def show_webb():
    df = data['pitcher_ag']
    webb_df = df[df['player'] == 'Webb'].copy()
    
    top4 = webb_df.reindex(webb_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    
    interpretation = """
**좌완 선발 Webb — SO vs Walk 분기 메커니즘** ⭐

5명 중 **가장 명확한 차이**가 발견된 케이스.

### 핵심 차이

- **HSS at FP**: SO 23.0° vs Walk 11.4° (p=0.005, d=2.16)
  - 삼진 시 상체-하체 분리가 2배 이상 큼
- **Trunk/Hip ratio**: SO 2.51 vs Walk 1.49 (p=0.005, d=3.05)
  - 삼진 시 몸통이 골반보다 훨씬 빠르게 회전 (kinetic chain)
- **Trunk peak 3D**: SO 1377 vs Walk 1134°/s
  - 회전력 자체가 더 큼

### 메커니즘 해석


### 시사점

- **폼 미세조정 가능 영역**: HSS 안정화 코칭 가치 있음
- 좌완 선발의 특성상 **타석에 대한 분리 메커니즘**이 결정적
- 코칭/Pre-pitch 루틴으로 일관성 확보 가능
"""
    
    show_pitcher_page("Webb", "SO", "Walk", interpretation, key_findings)


def show_garcia():
    df = data['pitcher_ag']
    garcia_df = df[df['player'] == 'Garcia'].copy()
    
    top4 = garcia_df.reindex(garcia_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    
    interpretation = """
**마무리 투수 Garcia — SV vs BS, Null Finding**

대부분 지표에서 통계적 유의성 없음 (p > 0.05).

### 의미

- Garcia의 **폼 자체는 매우 일관**됨
- Save와 Blown Save 차이가 **폼에서 비롯되지 않음**

### 가능한 BS 원인 (모션 외 요인)

1. **구속 저하**: 누적 등판 피로
2. **구질 효과**: 변화구 구사율 변화
3. **매치업**: 특정 타자/타순에 약점
4. **상황적 압박**: 9회 vs 연장 등

### 시사점

- 폼 교정으로 BS 줄이기 **어려움**
- **Deployment 변경** (등판 상황 조절)
- **매치업 specialist 활용**: 특정 타자 회피
- 또는 **외부 영입** 검토

본인 분석에서 GM의 외부 클로저 영입 결정과 일치.
"""
    
    show_pitcher_page("Garcia", "SV", "BS", interpretation, key_findings)


def show_armstrong():
    df = data['pitcher_ag']
    arm_df = df[df['player'] == 'Armstrong'].copy()
    
    top4 = arm_df.reindex(arm_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    
    interpretation = """
**불펜 투수 Armstrong — SV vs BS**

분석 결과 (Cohen's d 및 p-value 참고).

지표 차이의 정도와 통계적 유의성에 따라:

- **차이가 큰 지표**가 발견되면 → 폼 교정 영역
- **차이가 미미**하면 → Garcia처럼 외부 요인 의심

### 일반적 해석

불펜 투수는 짧은 등판이라 **워밍업 부족**으로 폼 변동이 클 수 있음.
일관성 향상을 위한 **루틴 표준화**가 도움이 될 가능성.
"""
    
    show_pitcher_page("Armstrong", "SV", "BS", interpretation, key_findings)


def show_jackson():
    df = data['pitcher_ag']
    jack_df = df[df['player'] == 'Jackson'].copy()
    
    top4 = jack_df.reindex(jack_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    
    interpretation = """
**불펜 투수 Jackson — SV vs BS, 해석**

**해석 **: **Lateral Trunk Tilt Overhand** 수치만 보면 Sidearm처럼 보이지만, 
실제로는 측면 굴곡이 있는 오버핸드 패턴으로 해석됨.    
- Trunk 3D / Trunk XZ ratio 1.2-1.3 → 측면 굴곡 magnitude
- Walker Buehler, Clay Holmes archetype에 가까움

### 시사점 — Form Correction 아닌 Deployment

폼은 그 자체로 **건강한 패턴** (sidearm처럼 부상 위험 X).
다만 좌타자/우타자 매치업 특성이 다를 수 있음.

**정책 함의**:
- 폼 교정 X
- **Specialist Deployment**: 특정 매치업에서 활용
- **Platoon split** 분석 → 강점 매치업 식별
"""
    
    show_pitcher_page("Jackson", "SV", "BS", interpretation, key_findings)

# ─────────────────────────────────────────────────────────────
# Page: Representative Pitcher Comparison
# ─────────────────────────────────────────────────────────────
def show_comparison():
    page_hero(
        "Comparison",
        "Representative Pitcher Motion Layer",
        "잔차 분석의 하위 근거로 선정한 5명 투수의 상황별 키네마틱 차이를 Cohen's d와 p-value 기준으로 비교합니다. 필요한 지표는 필터로 직접 확인할 수 있게 구성했습니다.",
        [("Cohen's d", "white"), ("p-value", "white"), ("Interactive", "white")],
    )

    df = data['pitcher_ag'].copy()
    available_metrics = df['label'].dropna().unique().tolist()
    available_players = df['player'].dropna().unique().tolist()

    def _sig_label(p_value: float) -> str:
        if pd.isna(p_value):
            return "n/a"
        if p_value < 0.001:
            return "p<0.001"
        if p_value < 0.01:
            return "p<0.01"
        if p_value < 0.05:
            return "p<0.05"
        if p_value < 0.10:
            return "p<0.10"
        return "n.s."

    def _sig_group(p_value: float) -> str:
        if pd.isna(p_value):
            return "No p-value"
        if p_value < 0.01:
            return "Strong"
        if p_value < 0.05:
            return "Significant"
        if p_value < 0.10:
            return "Marginal"
        return "Not significant"

    chart_col_1, chart_col_2 = st.columns([1.08, 1], gap="large")

    with chart_col_1:
        with st.container(border=True):
            st.markdown('<div class="chart-title">지표별 효과 크기</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-caption">선택한 지표에서 투수별 Cohen&#39;s d를 비교합니다. 점선은 |d|=0.8 기준입니다.</div>', unsafe_allow_html=True)
            selected_metric = st.selectbox(
                "지표 선택",
                available_metrics,
                index=available_metrics.index('HSS @ FP (°)') if 'HSS @ FP (°)' in available_metrics else 0,
                key="comparison_metric_select_compact",
            )
            selected_players_bar = st.multiselect(
                "표시할 투수",
                available_players,
                default=available_players,
                key="comparison_bar_players_compact",
            )
            metric_df = df[df['label'] == selected_metric].copy()
            if selected_players_bar:
                metric_df = metric_df[metric_df['player'].isin(selected_players_bar)].copy()
            metric_df['cohens_d'] = pd.to_numeric(metric_df['cohens_d'], errors='coerce')
            metric_df['u_p'] = pd.to_numeric(metric_df['u_p'], errors='coerce')
            metric_df['abs_d'] = metric_df['cohens_d'].abs()
            metric_df['sig_label'] = metric_df['u_p'].map(_sig_label)
            metric_df['sig_group'] = metric_df['u_p'].map(_sig_group)
            metric_df = metric_df.sort_values('abs_d', ascending=True)

            if metric_df.empty:
                st.info("선택한 조건에 해당하는 지표 데이터가 없습니다.")
            else:
                color_map = {
                    "Strong": TEX_RED,
                    "Significant": "#D04A52",
                    "Marginal": "#F59F00",
                    "Not significant": "#9AA4B2",
                    "No p-value": "#CBD5E1",
                }
                metric_df['bar_text'] = metric_df['cohens_d'].map(lambda v: f"{v:+.2f}")
                fig_effect = px.bar(
                    metric_df,
                    x='cohens_d',
                    y='player',
                    orientation='h',
                    color='sig_group',
                    color_discrete_map=color_map,
                    text='bar_text',
                    custom_data=['abs_d', 'u_p', 'a_mean', 'b_mean', 'diff', 'sig_label'],
                    labels={'cohens_d': "Cohen's d", 'player': "Pitcher", 'sig_group': "Significance"},
                )
                fig_effect.update_traces(
                    textposition='inside',
                    insidetextanchor='middle',
                    cliponaxis=False,
                    marker_line_color='rgba(255,255,255,0.9)',
                    marker_line_width=1.0,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        + f"Metric: {selected_metric}<br>"
                        + "Cohen's d: %{x:+.3f}<br>"
                        + "|d|: %{customdata[0]:.3f}<br>"
                        + "p-value: %{customdata[1]:.4f}<br>"
                        + "A mean: %{customdata[2]:.3f}<br>"
                        + "B mean: %{customdata[3]:.3f}<br>"
                        + "Diff: %{customdata[4]:+.3f}<br>"
                        + "Significance: %{customdata[5]}<extra></extra>"
                    ),
                )
                x_min = float(metric_df['cohens_d'].min())
                x_max = float(metric_df['cohens_d'].max())
                pad = max(0.35, (x_max - x_min) * 0.16)
                fig_effect.add_vline(x=0, line_width=1.2, line_color='rgba(13,27,51,0.48)')
                fig_effect.add_vline(x=0.8, line_width=1, line_dash='dash', line_color='rgba(13,27,51,0.25)')
                fig_effect.add_vline(x=-0.8, line_width=1, line_dash='dash', line_color='rgba(13,27,51,0.25)')
                fig_effect.update_layout(
                    height=318,
                    margin=dict(l=8, r=16, t=14, b=34),
                    xaxis=dict(title="Cohen's d", range=[x_min - pad, x_max + pad], zeroline=False, gridcolor='rgba(13,27,51,0.08)'),
                    yaxis=dict(title=None, categoryorder='array', categoryarray=metric_df['player'].tolist()),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='left', x=0, title=None, font=dict(size=10)),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family='Manrope, Arial, sans-serif', color='#1B2435', size=11),
                    bargap=0.36,
                )
                st.plotly_chart(fig_effect, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with chart_col_2:
        with st.container(border=True):
            st.markdown('<div class="chart-title">키네마틱 프로필 레이더</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-caption">선수별 라인만 표시합니다. 면 채움 없이 축별 분기 폭을 비교합니다.</div>', unsafe_allow_html=True)
            selected_players_radar = st.multiselect(
                "레이더 표시 투수",
                available_players,
                default=available_players,
                key="comparison_radar_players_compact",
            )
            selected_metrics_radar = st.multiselect(
                "레이더 지표",
                available_metrics,
                default=available_metrics,
                key="comparison_radar_metrics_compact",
            )
            radar_value_mode = st.radio(
                "값 기준",
                ["절대값 |d|", "부호 포함 d"],
                horizontal=True,
                key="comparison_radar_mode_compact",
            )

            radar_df = df[df['player'].isin(selected_players_radar) & df['label'].isin(selected_metrics_radar)].copy()
            radar_df['cohens_d'] = pd.to_numeric(radar_df['cohens_d'], errors='coerce')
            radar_df['u_p'] = pd.to_numeric(radar_df['u_p'], errors='coerce')
            radar_df['radar_value'] = radar_df['cohens_d'].abs() if radar_value_mode == "절대값 |d|" else radar_df['cohens_d']
            radar_df['sig_label'] = radar_df['u_p'].map(_sig_label)

            if radar_df.empty or not selected_metrics_radar or not selected_players_radar:
                st.info("레이더 차트를 그리려면 투수와 지표를 1개 이상 선택해 주세요.")
            else:
                metric_order = selected_metrics_radar
                radar_palette = [TEX_BLUE, TEX_RED, '#2F9E65', '#F59F00', '#7C3AED', '#0EA5E9', '#64748B']
                fig_radar = go.Figure()
                for idx, player in enumerate(selected_players_radar):
                    player_frame = radar_df[radar_df['player'] == player].set_index('label')
                    values, hover_lines = [], []
                    for metric in metric_order:
                        if metric in player_frame.index:
                            row = player_frame.loc[metric]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            value = float(row['radar_value']) if pd.notna(row['radar_value']) else 0.0
                            original_d = float(row['cohens_d']) if pd.notna(row['cohens_d']) else 0.0
                            p_value = float(row['u_p']) if pd.notna(row['u_p']) else np.nan
                            hover_lines.append(f"{metric}<br>d: {original_d:+.3f}<br>|d|: {abs(original_d):.3f}<br>p: {p_value:.4f}")
                        else:
                            value = 0.0
                            hover_lines.append(f"{metric}<br>No data")
                        values.append(value)
                    closed_metrics = metric_order + [metric_order[0]]
                    closed_values = values + [values[0]]
                    closed_hover = hover_lines + [hover_lines[0]]
                    color = radar_palette[idx % len(radar_palette)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=closed_values,
                        theta=closed_metrics,
                        mode='lines+markers',
                        name=player,
                        line=dict(color=color, width=2.6),
                        marker=dict(size=5, color=color),
                        fill=None,
                        opacity=1.0,
                        customdata=closed_hover,
                        hovertemplate="<b>%{fullData.name}</b><br>%{customdata}<extra></extra>",
                    ))
                max_abs = float(radar_df['radar_value'].abs().max()) if not radar_df.empty else 1.0
                radial_max = max(1.0, max_abs * 1.12)
                radial_min = -radial_max if radar_value_mode == "부호 포함 d" else 0
                fig_radar.update_layout(
                    height=360,
                    margin=dict(l=8, r=8, t=8, b=8),
                    polar=dict(
                        bgcolor='rgba(255,255,255,0)',
                        radialaxis=dict(visible=True, range=[radial_min, radial_max], gridcolor='rgba(13,27,51,0.10)', linecolor='rgba(13,27,51,0.18)', tickfont=dict(size=9, color='#64748B')),
                        angularaxis=dict(gridcolor='rgba(13,27,51,0.08)', linecolor='rgba(13,27,51,0.14)', tickfont=dict(size=9, color='#1B2435')),
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='left', x=0, title=None, font=dict(size=10)),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family='Manrope, Arial, sans-serif', color='#1B2435', size=11),
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    if not df.empty:
        summary_df = (
            df.assign(abs_d=pd.to_numeric(df['cohens_d'], errors='coerce').abs(), u_p=pd.to_numeric(df['u_p'], errors='coerce'))
            .groupby('player', as_index=False)
            .agg(avg_abs_d=('abs_d', 'mean'), max_abs_d=('abs_d', 'max'), sig_count=('u_p', lambda s: int((s < 0.05).sum())))
            .sort_values(['avg_abs_d', 'max_abs_d'], ascending=False)
        )
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, (_, row) in zip([s1, s2, s3, s4, s5], summary_df.iterrows()):
            with col:
                kpi_card(str(row['player']), fmt_num(row['avg_abs_d'], 2), f"max |d| {fmt_num(row['max_abs_d'], 2)} · sig {int(row['sig_count'])}", accent="red" if int(row['sig_count']) > 0 else "navy")

    section_header("투수별 종합 요약", "모든 지표를 표로 비교합니다. 진한 색일수록 효과 크기 또는 유의성이 큽니다.")
    pivot_d = df.pivot_table(index='player', columns='label', values='cohens_d').round(2)
    pivot_p = df.pivot_table(index='player', columns='label', values='u_p').round(3)

    def color_d(val):
        if pd.isna(val):
            return ''
        abs_d = abs(val)
        if abs_d > 1.5:
            return 'background-color: #003278; color: white; font-weight: bold'
        if abs_d > 0.8:
            return 'background-color: #6080B0; color: white'
        if abs_d > 0.5:
            return 'background-color: #B8C8E0'
        return ''

    def color_p(val):
        if pd.isna(val):
            return ''
        if val < 0.01:
            return 'background-color: #C0111F; color: white; font-weight: bold'
        if val < 0.05:
            return 'background-color: #E08080'
        if val < 0.10:
            return 'background-color: #FFE4B5'
        return ''

    table_tab_1, table_tab_2 = st.tabs(["Cohen's d", "p-value"])
    with table_tab_1:
        st.dataframe(pivot_d.style.map(color_d).format("{:+.2f}", na_rep="—"), use_container_width=True)
        st.caption("진한 파랑: |d|>1.5 | 파랑: |d|>0.8 | 연파랑: |d|>0.5")
    with table_tab_2:
        st.dataframe(pivot_p.style.map(color_p).format("{:.3f}", na_rep="—"), use_container_width=True)
        st.caption("빨강: p<0.01 | 연빨강: p<0.05 | 연주황: p<0.10")

    section_header("투수별 핵심 발견", "폼 교정이 필요한 선수와 운영·매치업 관점으로 봐야 하는 선수를 구분합니다.")
    summary_cards = [
        ("Leiter", "선발", "ns 대부분", "폼 자체는 일관, 결과 차이는 다른 요인", "🟡"),
        ("Webb", "선발", "강한 차이 ⭐", "HSS, Trunk/Hip ratio에서 매우 명확", "🔴"),
        ("Garcia", "마무리", "Null finding", "폼 일관됨, BS는 외부 요인 가능성", "🟡"),
        ("Armstrong", "불펜", "지표 변동", "p<0.05 일부 지표 존재", "🟠"),
        ("Jackson", "불펜", "재해석", "Sidearm 아닌 Lateral tilt overhand", "🟢"),
    ]
    html = '<div class="inline-stat-grid">'
    for name, role, finding, note, marker in summary_cards:
        html += (
            f'<div class="pitcher-mini-card">'
            f'<div class="pitcher-mini-name">{marker} {name}</div>'
            f'<div class="pitcher-mini-role">{role}</div>'
            f'<div class="pitcher-mini-tag">{finding}</div>'
            f'<div class="pitcher-mini-note">{note}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-shell glass-card-accent"><div class="section-heading">폼 교정 가능 영역</div><div class="section-copy">Webb: HSS 안정화 코칭<br>Armstrong: 등판 루틴 표준화<br><br><b>→ 코칭 스태프 개입 가치</b></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-shell"><div class="section-heading">폼 외 요인 의심</div><div class="section-copy">Leiter: 피칭 디자인 검토<br>Garcia: Deployment / 매치업<br>Jackson: Specialist 활용<br><br><b>→ 운영 차원 결정</b></div></div>', unsafe_allow_html=True)



# ─────────────────────────────────────────────────────────────
# Page: Simulation
# ─────────────────────────────────────────────────────────────
def show_simulation():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">Monte Carlo</span>
        <span class="pill pill-white">Residual Model</span>
        <span class="pill pill-white">2025 Season Rebuild</span>
        <h1>2025 TEX 시즌 시뮬레이션</h1>
        <p>
        2025 텍사스 레인저스의 실제 시즌을 기준으로 주요 전력 변수 변화가 승수에 미치는 영향을 재구성합니다.
        경기력 분석과 선수 분석에서 나온 가설을 가상 시나리오로 돌렸을 때 승수 분포와 residual layer가 어떻게 바뀌는지 확인합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_badges(
        ("Baseline / Bullpen / Rotation / Hitter Scenario", "red"),
        ("Actual 162-game Schedule", "navy"),
    )

    st.markdown("---")

    control_left, control_right = st.columns([1.2, 1], gap="large")

    with control_left:
        st.markdown("## 시나리오 실행")
        st.markdown("""
        <div class="glass-card">
            <b>실행 방식</b><br>
            시뮬레이션은 페이지 최초 진입 시 자동 실행되지 않습니다. 아래 조건을 선택한 뒤 버튼을 누르면
            <code>simulator.py</code>의 <code>run_simulation()</code>을 호출해 경기력·선수 조건 변화가 잔차와 승수에 미치는 결과를 계산합니다.
        </div>
        """, unsafe_allow_html=True)

    with control_right:
        st.markdown("## 조건 선택")
        selected_scenario = st.selectbox(
            "Scenario",
            SIMULATION_OPTIONS,
            index=SIMULATION_OPTIONS.index(st.session_state.get("sim_scenario", DEFAULT_SCENARIO)),
            key="simulation_scenario_select",
        )
        simulation_runs = st.slider(
            "Monte Carlo Runs",
            min_value=100,
            max_value=1000,
            value=int(st.session_state.get("sim_runs", DEFAULT_SIM_RUNS)),
            step=100,
            key="simulation_runs_slider",
        )
        run_click = st.button("Run Simulation", type="primary", use_container_width=True)

    if "simulation_result" not in st.session_state:
        st.session_state["simulation_result"] = None
        st.session_state["sim_scenario"] = DEFAULT_SCENARIO
        st.session_state["sim_runs"] = DEFAULT_SIM_RUNS

    if run_click:
        if not RAW_DIR.exists():
            st.error("data_raw 폴더가 없습니다. app.py/simulator.py에서 쓰던 CSV 파일들을 data_raw 폴더에 넣어주세요.")
            return
        try:
            with st.spinner("시뮬레이션 실행 중..."):
                st.session_state["simulation_result"] = get_simulation_result(
                    str(RAW_DIR),
                    selected_scenario,
                    simulation_runs,
                )
                st.session_state["sim_scenario"] = selected_scenario
                st.session_state["sim_runs"] = simulation_runs
        except FileNotFoundError as exc:
            st.error(f"필수 데이터 파일이 없습니다: {exc}")
            st.info("data_raw 폴더 안의 CSV 파일명과 simulator.py에서 요구하는 파일명이 같은지 확인해주세요.")
            return
        except Exception as exc:
            st.error(f"시뮬레이션 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
            return

    result = st.session_state.get("simulation_result")

    if result is None:
        finding_box(
            "아직 시뮬레이션을 실행하지 않았습니다.",
            "상단에서 시나리오와 반복 횟수를 선택한 뒤 Run Simulation을 눌러주세요. 앱 첫 로딩 속도를 위해 계산은 lazy-load 방식으로 분리했습니다."
        )
        return

    summary = result.get("summary", {})
    distribution = result.get("distribution", pd.DataFrame()).copy()
    monthly = result.get("monthly_summary", pd.DataFrame()).copy()
    schedule_context = result.get("schedule_context", pd.DataFrame()).copy()
    players = result.get("player_projection", pd.DataFrame()).copy()
    pitchers = result.get("pitcher_projection", pd.DataFrame()).copy()

    st.markdown("---")
    st.markdown("## 시뮬레이션 결과 요약")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Mean Wins", fmt_num(summary.get("mean")), f"Scenario: {st.session_state['sim_scenario']}", accent="red")
    with k2:
        kpi_card("Median Wins", fmt_num(summary.get("median")), f"Runs: {st.session_state['sim_runs']}", accent="navy")
    with k3:
        kpi_card("P10 - P90", f"{fmt_num(summary.get('p10'))} - {fmt_num(summary.get('p90'))}", "중앙 80% 결과 범위", accent="red")
    with k4:
        kpi_card("82+ Wins", fmt_pct(summary.get("over_81_5")), "0.500 이상 가능성", accent="navy")

    st.markdown("""
    <div class="finding-box">
        <strong>해석 기준</strong><br>
        이 결과는 미래 예측이라기보다 <b>2025 시즌을 조건 변화에 따라 다시 재구성한 결과</b>입니다.
        피타고리안 기대 승수와 실제 승수의 괴리를 설명하기 위해, 시나리오별 승수 분포와 residual 보정값을 함께 확인합니다.
    </div>
    """, unsafe_allow_html=True)

    chart_left, chart_right = st.columns([1.35, 1], gap="large")

    with chart_left:
        st.markdown("### 승수 분포")
        if not distribution.empty and "wins" in distribution.columns:
            win_chart = (
                alt.Chart(distribution)
                .mark_bar(color="#B31922", opacity=0.82, binSpacing=2)
                .encode(
                    x=alt.X("wins:Q", bin=alt.Bin(maxbins=18), title="Projected Wins"),
                    y=alt.Y("count():Q", title="Simulation Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
                .properties(height=330)
            )
            st.altair_chart(win_chart, use_container_width=True)
        else:
            st.info("승수 분포 데이터가 없습니다.")

    with chart_right:
        st.markdown("### Residual Layer")
        st.markdown(f"""
        <div class="glass-card glass-card-accent">
            <div class="kpi-label-custom">Active Scenario</div>
            <div class="kpi-value-custom" style="font-size:22px;">{st.session_state['sim_scenario']}</div>
            <div class="kpi-sub-custom">Monte Carlo {st.session_state['sim_runs']}회</div>
            <hr style="margin:14px 0;">
            <div style="font-size:13px; line-height:1.8; color:#344054;">
                <b>Residual bonus</b>: <span class="num">{fmt_num(summary.get('residual_bonus'), 2, sign=True)}</span> wins<br>
                <b>CV MAE</b>: <span class="num">{fmt_num(summary.get('ensemble_cv_mae'), 2)}</span><br>
                <b>Calibration offset</b>: <span class="num">{fmt_num(summary.get('calibration_offset'), 2, sign=True)}</span> wins<br>
                <b>88+ Wins</b>: <span class="num">{fmt_pct(summary.get('over_87_5'))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 월별 시뮬레이션 흐름")
    monthly_left, monthly_right = st.columns([1.35, 1], gap="large")

    with monthly_left:
        if not monthly.empty and {"month", "mean_wins", "p25_wins", "p75_wins"}.issubset(monthly.columns):
            month_order = list(monthly["month"])
            monthly_band = (
                alt.Chart(monthly)
                .mark_area(opacity=0.18, color="#B31922")
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("p25_wins:Q", title="Projected Wins"),
                    y2="p75_wins:Q",
                    tooltip=[
                        alt.Tooltip("month:N", title="Month"),
                        alt.Tooltip("mean_wins:Q", title="Mean", format=".2f"),
                        alt.Tooltip("p25_wins:Q", title="P25", format=".2f"),
                        alt.Tooltip("p75_wins:Q", title="P75", format=".2f"),
                    ],
                )
            )
            monthly_line = (
                alt.Chart(monthly)
                .mark_line(point=True, strokeWidth=3, color="#0D1B33")
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("mean_wins:Q", title="Projected Wins"),
                )
            )
            st.altair_chart((monthly_band + monthly_line).properties(height=300), use_container_width=True)
        else:
            st.info("월별 요약 데이터가 없습니다.")

    with monthly_right:
        st.markdown("### Schedule Context")
        if not schedule_context.empty:
            rename_map = {
                "month": "Month",
                "games": "G",
                "home_games": "Home",
                "away_games": "Away",
                "strength_index": "Strength",
                "difficulty": "Difficulty",
            }
            display = schedule_context.rename(columns=rename_map)
            keep_cols = [col for col in ["Month", "G", "Home", "Away", "Strength", "Difficulty"] if col in display.columns]
            st.dataframe(display[keep_cols], use_container_width=True, hide_index=True)
        else:
            st.info("스케줄 요약 데이터가 없습니다.")

    st.markdown("---")
    hitter_tab, pitcher_tab, scenario_tab = st.tabs(["Hitters", "Pitchers", "Scenario Board"])

    with hitter_tab:
        st.markdown("### 타자 시나리오 카드")
        if not players.empty:
            card_players = players.head(4).copy()
            cols = st.columns(4)
            for col, (_, player) in zip(cols, card_players.iterrows()):
                with col:
                    kpi_card(
                        str(player.get("player", "-")),
                        fmt_num(player.get("sim_on_base"), 3),
                        f"{player.get('archetype', '-')} | HR {fmt_num(player.get('sim_hr'), 3)}",
                        accent="red" if bool(player.get("boosted", False)) else "navy",
                    )

            player_display = players.copy()
            for source, target, digits in [
                ("sim_on_base", "Modeled OBP", 3),
                ("sim_hr", "HR Rate", 3),
                ("sim_xbh", "XBH Rate", 3),
                ("sim_k", "K Rate", 3),
                ("delta_obp_pts", "OBP Delta", 1),
                ("delta_hr_pts", "HR Delta", 1),
            ]:
                if source in player_display.columns:
                    player_display[target] = player_display[source].map(lambda v: fmt_num(v, digits, sign=("Delta" in target)))
            keep = [c for c in ["player", "pos", "archetype", "Modeled OBP", "HR Rate", "XBH Rate", "K Rate", "OBP Delta", "HR Delta"] if c in player_display.columns]
            st.dataframe(player_display[keep].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("타자 projection 데이터가 없습니다.")

    with pitcher_tab:
        st.markdown("### 투수 시나리오 카드")
        if not pitchers.empty:
            card_pitchers = pitchers.head(4).copy()
            cols = st.columns(4)
            for col, (_, pitcher) in zip(cols, card_pitchers.iterrows()):
                delta_era = float(pitcher.get("delta_era", 0.0)) if pd.notna(pitcher.get("delta_era", 0.0)) else 0.0
                with col:
                    kpi_card(
                        str(pitcher.get("player", "-")),
                        fmt_num(pitcher.get("sim_era"), 2),
                        f"{pitcher.get('role', '-')} | ΔERA {fmt_num(delta_era, 2, sign=True)}",
                        accent="navy" if delta_era <= 0 else "red",
                    )

            pitcher_display = pitchers.copy()
            for source, target, digits in [
                ("sim_era", "Modeled ERA", 2),
                ("sim_whip", "Modeled WHIP", 2),
                ("sim_k9", "Modeled K/9", 1),
                ("delta_era", "ERA Delta", 2),
                ("sim_ip", "Projected IP", 1),
            ]:
                if source in pitcher_display.columns:
                    pitcher_display[target] = pitcher_display[source].map(lambda v: fmt_num(v, digits, sign=(target == "ERA Delta")))
            keep = [c for c in ["player", "role", "archetype", "Modeled ERA", "Modeled WHIP", "Modeled K/9", "ERA Delta", "Projected IP"] if c in pitcher_display.columns]
            st.dataframe(pitcher_display[keep].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("투수 projection 데이터가 없습니다.")

    with scenario_tab:
        st.markdown("### 시나리오별 선수 변화 비교")
        try:
            snapshots = get_scenario_snapshots(str(RAW_DIR))
        except Exception as exc:
            st.warning(f"시나리오 보드 데이터를 불러오지 못했습니다: {type(exc).__name__}: {exc}")
            snapshots = None

        if snapshots:
            compare_mode = st.radio("Comparison Type", ["Hitters", "Pitchers"], horizontal=True)
            if compare_mode == "Hitters" and "hitters" in snapshots:
                hitter_snapshots = snapshots["hitters"].copy()
                hitter_names = hitter_snapshots["player"].drop_duplicates().tolist()
                default_idx = hitter_names.index("Wyatt Langford") if "Wyatt Langford" in hitter_names else 0
                selected_hitter = st.selectbox("Player", hitter_names, index=default_idx)
                hitter_compare = hitter_snapshots[hitter_snapshots["player"] == selected_hitter].copy()
                compare_chart = (
                    alt.Chart(hitter_compare)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#B31922")
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_on_base:Q", title="Modeled OBP"),
                        tooltip=["scenario:N", alt.Tooltip("sim_on_base:Q", format=".3f"), alt.Tooltip("sim_hr:Q", format=".3f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.dataframe(hitter_compare, use_container_width=True, hide_index=True)
            elif compare_mode == "Pitchers" and "pitchers" in snapshots:
                pitcher_snapshots = snapshots["pitchers"].copy()
                pitcher_names = pitcher_snapshots["player"].drop_duplicates().tolist()
                default_idx = pitcher_names.index("Nathan Eovaldi") if "Nathan Eovaldi" in pitcher_names else 0
                selected_pitcher = st.selectbox("Pitcher", pitcher_names, index=default_idx)
                pitcher_compare = pitcher_snapshots[pitcher_snapshots["player"] == selected_pitcher].copy()
                compare_chart = (
                    alt.Chart(pitcher_compare)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#0D1B33")
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_era:Q", title="Modeled ERA"),
                        tooltip=["scenario:N", alt.Tooltip("sim_era:Q", format=".2f"), alt.Tooltip("delta_era:Q", format=".2f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.dataframe(pitcher_compare, use_container_width=True, hide_index=True)
            else:
                st.info("선택한 비교 데이터가 없습니다.")
        else:
            st.info("시뮬레이션 실행 결과는 표시되지만, scenario snapshot 데이터는 아직 불러오지 못했습니다.")


# ─────────────────────────────────────────────────────────────
# Page: Conclusions
# ─────────────────────────────────────────────────────────────
def show_conclusions():
    page_hero(
        "Conclusions",
        "Residual Analysis Takeaways",
        "경기력 분석, 선수 분석, 대표 투수 모션 분석, 시나리오 시뮬레이션, AI Agent 질의를 종합해 2025 TEX 잔차 -9.06승 해석과 보고서 출력으로 마무리합니다.",
        [("Residual Summary", "white"), ("Player Reports", "white"), ("Team PDF", "white")],
    )

    finding_box(
        "종합 흐름",
        "이 페이지는 모션 분석만의 결론이 아니라, 실제 81승과 Pythagorean 90.06승 사이의 -9.06승 잔차를 설명하기 위한 최종 요약입니다. "
        "대표 투수 모션 결과는 경기력·선수·시뮬레이션 분석을 보완하는 세부 근거로 해석합니다."
    )

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown('''
        <div class="section-shell glass-card-accent">
            <div class="section-heading">폼 차이 명확 · 코칭 가능</div>
            <div class="section-copy">
                <b>Webb</b>는 HSS와 Trunk/Hip ratio에서 강한 차이를 보였습니다.<br><br>
                SO 시 상하체 분리와 kinetic chain이 안정적이고, Walk 시에는 동시 회전에 가까워지며 정확성이 흔들리는 흐름입니다.
            </div>
        </div>
        ''', unsafe_allow_html=True)
    with col2:
        st.markdown('''
        <div class="section-shell">
            <div class="section-heading">폼 차이 미미 · 외부 요인</div>
            <div class="section-copy">
                <b>Garcia</b>와 <b>Leiter</b>는 폼 자체보다 구질, 위치, 카운트 운영, 매치업 영향이 더 클 가능성이 있습니다.<br><br>
                따라서 전원 폼 교정보다는 선수별 원인 분리가 필요합니다.
            </div>
        </div>
        ''', unsafe_allow_html=True)

    st.markdown('''
    <div class="section-shell compact">
        <div class="section-heading">재해석 케이스 · Jackson</div>
        <div class="section-copy">
            Jackson은 sidearm이라기보다 <b>Lateral Tilt Overhand</b>에 가까운 패턴입니다. 폼 자체를 부상 위험으로 보기보다, platoon split과 specialist deployment 관점으로 해석하는 편이 적절합니다.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    section_header("투수진 운영 권고", "분석 결과를 코칭, 피칭 디자인, deployment 관점으로 연결했습니다.")
    rec_df = pd.DataFrame({
        '투수': ['Leiter', 'Webb ⭐', 'Garcia', 'Armstrong', 'Jackson'],
        '발견': ['ns (폼 일관)', 'HSS, Ratio p<0.01', 'Null finding', '일부 지표 변동', 'Lateral tilt overhand'],
        '권고': ['피칭 디자인 검토', '⭐ HSS 안정화 코칭', 'Deployment 조절 / 외부 영입', '등판 전 루틴 표준화', 'Platoon specialist 활용'],
        '우선순위': ['중', '높', '높', '중', '중'],
    })
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown('''
    <div class="section-shell glass-card-accent">
        <div class="section-heading">GM 결정과의 연결</div>
        <div class="section-copy">
            본 모션 분석은 선발 보강, 외부 클로저 검토, 불펜 재편 같은 프론트오피스 의사결정과 논리적으로 맞물립니다.
            특히 Garcia의 null finding은 폼 교정보다 deployment나 외부 보강의 필요성을 뒷받침합니다.
        </div>
    </div>
    ''', unsafe_allow_html=True)

    with st.expander("📖 분석 한계 및 향후 과제", expanded=False):
        st.markdown('''
        ### 데이터 한계
        - Sample size: 선수당 5-7 trials → 통계 검정력 제한
        - Camera angle: 단일 시점 broadcast feed → 3D 정확도 제한
        - FPS: 약 59 fps → 빠른 피크 회전 일부 누락 가능

        ### 모델 한계
        - MotionAGFormer는 야구 특화 모델이 아님
        - Bone length normalization을 적용했지만 절대값 정확도에는 한계

        ### 향후 개선 방향
        - Pitch type별 별도 분석
        - Season-long longitudinal tracking
        - 다중 카메라 기반 triangulation
        - 야구 특화 pose model 학습
        ''')

    st.markdown('''
    <div class="quote-card">
        <p><strong>종합 결론</strong></p>
        <p>
        TEX 2025 잔차 -9.06승은 단일 원인으로 설명하기 어렵습니다. 모션 분석에서는 5명 중 Webb만 명확한 폼 분기를 보였고,
        나머지는 폼 외 요인이 더 중요했습니다. 따라서 경기 운영, 선수 상태, 투수별 모션 케이스, 시뮬레이션 결과를 함께 묶은 맞춤형 해석이 필요합니다.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("---")
    section_header("보고서 출력", "최종 발표/공유용으로 선수별 요약 보고서와 팀 종합 보고서를 PDF로 다운로드합니다.")
    report_col_1, report_col_2 = st.columns([1, 1], gap="large")
    with report_col_1:
        selected_report_player = st.selectbox(
            "선수별 보고서 대상",
            list(REPORT_FINDINGS.keys()),
            key="player_report_select",
        )
        st.download_button(
            "선수별 보고서 PDF 출력",
            data=build_player_report_pdf(selected_report_player),
            file_name=f"tex_2025_player_report_{selected_report_player.lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with report_col_2:
        st.markdown("""
        <div class="glass-card">
            <b>팀 요약 보고서</b><br>
            잔차 분석 목적, 분석 흐름, 대표 투수별 핵심 해석, 최종 결론을 한 문서로 정리합니다.
        </div>
        """, unsafe_allow_html=True)
        st.download_button(
            "팀 요약 보고서 PDF 출력",
            data=build_team_report_pdf(),
            file_name="tex_2025_team_residual_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    st.caption("📊 Dashboard by 지소윤 | 2026 Final Baseball Project")


def show_ai_agent():
    page_hero(
        "AI Agent",
        "CSV 기반 분석 질의",
        "업로드된 CSV 파일을 기반으로 사용자가 궁금해하는 TEX 2025 잔차 분석 질문에 답합니다. 텍스트 요약뿐 아니라 그래프로 확인할 만한 구조화 결과도 함께 탐색하는 역할입니다.",
        [("Text Q&A", "white"), ("Chart-ready Data", "white"), ("Uploaded CSV", "white")],
    )
    try:
        from agent import chatbot
    except ModuleNotFoundError as exc:
        st.error(f"AI Agent 의존성을 불러오지 못했습니다: {exc}")
        st.info("필요 패키지: `pydantic-ai`, `python-dotenv`")
        return
    except Exception as exc:
        st.error(f"AI Agent 초기화 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        return

    chatbot.render()



# ─────────────────────────────────────────────────────────────
# 페이지 라우팅
# ─────────────────────────────────────────────────────────────
if page == "overview":
    show_overview()
elif page == "simulation":
    show_simulation()
elif page == "methodology":
    show_methodology()
elif page == "leiter":
    show_leiter()
elif page == "webb":
    show_webb()
elif page == "garcia":
    show_garcia()
elif page == "armstrong":
    show_armstrong()
elif page == "jackson":
    show_jackson()
elif page == "comparison":
    show_comparison()
elif page == "ai_agent":
    show_ai_agent()
elif page == "conclusions":
    show_conclusions()
