"""TEX 2025 Pythagorean Residual Analysis Dashboard."""
# ── 한글 폰트 설정 ── Streamlit 명령어보다 먼저 실행해야 함 ──
import platform as _platform
import sys as _sys
import glob as _glob
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm


def _setup_korean_font():
    _os = _platform.system()
    if _os == 'Darwin':
        plt.rcParams['font.family'] = 'AppleGothic'
    elif _os == 'Windows':
        try:
            _pkg = _sys.prefix + "/Lib/site-packages/koreanize_matplotlib/fonts"
            for _f in _glob.glob(_pkg + "/*.ttf"):
                _fm.fontManager.addfont(_f)
            _fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        plt.rcParams['font.family'] = ['Malgun Gothic', 'NanumGothic', 'DejaVu Sans']
    else:
        try:
            _pkg = _sys.prefix + "/lib/python*/site-packages/koreanize_matplotlib/fonts"
            for _f in _glob.glob(_pkg + "/*.ttf"):
                _fm.fontManager.addfont(_f)
            _fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass
        plt.rcParams['font.family'] = ['NanumGothic', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42


_setup_korean_font()

# ── Streamlit 페이지 설정 ── 첫 번째 st.* 호출이어야 함 ───────────
import streamlit as st

st.set_page_config(
    page_title="TEX 2025 Residual Analysis",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공유 유틸리티 및 뷰 모듈 임포트 (set_page_config 이후) ─────────
from shared import ASSETS, image_to_base64
import views.overview as v_overview
import views.simulation as v_simulation
import views.methodology as v_methodology
import views.pitcher as v_pitcher          # noqa: F401  (사이드 이펙트용 임포트)
import views.leiter as v_leiter
import views.webb as v_webb
import views.garcia as v_garcia
import views.armstrong as v_armstrong
import views.jackson as v_jackson
import views.comparison as v_comparison
import views.ai_agent as v_ai_agent
import views.conclusions as v_conclusions

_APP_DIR = Path(__file__).resolve().parent
_V5_OUTPUT_FILES = {
    "Pareto": _APP_DIR / "output" / "pareto_summary.csv",
    "Grid Pareto": _APP_DIR / "output" / "grid_pareto.csv",
    "Decision Leaderboard": _APP_DIR / "output" / "scenario_decision_leaderboard.csv",
}

# ── 전역 CSS ────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url("https://fonts.googleapis.com/css2?family=Sora:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");
@import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

/* ─────────────────────────────
   테마 변수
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
   기본 설정
───────────────────────────── */
html, body, [class*="css"] {
    font-family: "Manrope", "Pretendard", "Noto Sans KR", system-ui, sans-serif;
}

.bi,
i[class^="bi-"],
i[class*=" bi-"] {
    font-family: "bootstrap-icons" !important;
    font-style: normal !important;
    font-weight: normal !important;
    line-height: 1 !important;
    display: inline-block;
}

[data-testid="stExpander"] summary p {
    font-family: "Manrope", "Pretendard", "Noto Sans KR", system-ui, sans-serif !important;
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

span[data-testid="stIconMaterial"],
.material-icons,
.material-icons-round,
.material-icons-rounded,
.material-symbols-outlined,
.material-symbols-rounded,
.material-symbols-sharp {
    font-family: "Material Symbols Rounded", "Material Icons Round", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    font-feature-settings: "liga" !important;
    -webkit-font-feature-settings: "liga" !important;
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

[data-testid="stHeader"] {
    background: transparent !important;
}

[data-testid="stMultiSelect"] label,
[data-testid="stMultiSelect"] span,
[data-testid="stMultiSelect"] input {
    font-size: 13px !important;
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
    font-family: "Manrope", "Pretendard", "Noto Sans KR", system-ui, sans-serif !important;
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
   사이드바 스타일
───────────────────────────── */
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

section[data-testid="stSidebar"] [data-testid="stSidebarHeader"],
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] > div,
section[data-testid="stSidebarContent"],
section[data-testid="stSidebarUserContent"] {
    padding: 0 !important;
    margin: 0 !important;
    background: transparent !important;
    overflow-x: hidden !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6 {
    color: #FFFFFF !important;
    font-family: "Sora", "Manrope", "Noto Sans KR", sans-serif !important;
    font-weight: 800 !important;
    letter-spacing: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.sidebar-brand-shell),
section[data-testid="stSidebar"] .element-container:has(.sidebar-sticky-head),
section[data-testid="stSidebar"] .stMarkdown:has(.sidebar-brand-shell),
section[data-testid="stSidebar"] .stMarkdown:has(.sidebar-sticky-head),
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.sidebar-brand-shell),
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"]:has(.sidebar-sticky-head) {
    width: 100% !important;
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.sidebar-sticky-head) {
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
    background: linear-gradient(
        to bottom,
        #FFFFFF 0px,
        #FFFFFF 118px,
        #071A35 118px,
        #071A35 100%
    ) !important;
}

.sidebar-sticky-head {
    position: sticky;
    top: 0;
    z-index: 100;
    width: 100%;
    background: linear-gradient(
        to bottom,
        #FFFFFF 0px,
        #FFFFFF 118px,
        #071A35 118px,
        #071A35 100%
    );
}

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

.sidebar-project-wrap {
    padding: 22px 18px 16px 18px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 0;
}

.sidebar-section-label {
    color: rgba(255,255,255,0.58);
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 9px;
}

.sidebar-project-title {
    color: #FFFFFF;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 19px;
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

.sidebar-note-wrap .sidebar-section-label {
    margin-bottom: 6px;
}

.sidebar-note-body {
    color: rgba(255,255,255,0.68);
    font-size: 13px;
    line-height: 1.55;
    text-align: left;
}

.sidebar-note-body .note-line {
    display: block;
}

.sidebar-note-body .note-subline {
    display: block;
    margin-top: 6px;
}

/* ─────────────────────────────
   히어로
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
   필(Pill)
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

.pill-navy-solid {
    background: #003278;
    color: #fff;
    border-color: #003278;
}

.pill-card-navy {
    background: rgba(240,245,255,0.97);
    color: #003278;
    border-color: rgba(0,50,120,0.22);
}

.pill-card-red {
    background: rgba(255,245,245,0.97);
    color: #B3191A;
    border-color: rgba(179,25,34,0.22);
}

.pill-gray {
    background: rgba(100,116,139,0.10);
    color: #475569;
    border-color: rgba(100,116,139,0.22);
}

.pill-white {
    background: rgba(255,255,255,0.13);
    color: white;
    border-color: rgba(255,255,255,0.20);
}

/* ─────────────────────────────
   글래스 카드
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
    margin-bottom: 24px;
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
    margin: 4px 0 20px 0;
    font-size: 14px;
    line-height: 1.65;
}

.glass-card .section-heading {
    margin-bottom: 8px !important;
}

.glass-card .section-copy {
    margin-bottom: 0 !important;
}

.glass-card .chart-caption,
.glass-card .chart-title {
    margin-bottom: 0 !important;
}

.glass-card .chart-caption {
    font-size: 14px;
    line-height: 1.65;
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

.glass-card-red {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(179,25,34,0.12) 0%, transparent 55%),
        linear-gradient(145deg, rgba(255,245,245,0.95), rgba(255,235,235,0.80));
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid rgba(179,25,34,0.22);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 10px 28px -14px rgba(179,25,34,0.30),
        0 2px 6px -2px rgba(13,27,51,0.10);
}

.glass-card-navy {
    background:
        radial-gradient(120% 80% at 0% 0%, rgba(0,50,120,0.10) 0%, transparent 55%),
        linear-gradient(145deg, rgba(240,245,255,0.95), rgba(225,235,255,0.80));
    backdrop-filter: blur(14px) saturate(140%);
    -webkit-backdrop-filter: blur(14px) saturate(140%);
    border: 1px solid rgba(0,50,120,0.20);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 10px 28px -14px rgba(0,50,120,0.22),
        0 2px 6px -2px rgba(13,27,51,0.10);
}

.glass-card-amber {
    background:
        linear-gradient(145deg, rgba(241,243,246,0.97), rgba(226,230,235,0.85));
    backdrop-filter: blur(14px) saturate(120%);
    -webkit-backdrop-filter: blur(14px) saturate(120%);
    border: 1px solid rgba(100,110,130,0.20);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 10px 28px -14px rgba(100,110,130,0.20),
        0 2px 6px -2px rgba(13,27,51,0.08);
}

/* KPI 카드 */
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
    margin-bottom: 24px !important;
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

/* 핵심 발견 박스 */
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

.finding-box-navy {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(13,27,51,0.09) 0%, transparent 55%),
        linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,255,255,0.72));
    border: 1px solid rgba(13,27,51,0.16);
    color: #344054;
    border-radius: 18px;
    padding: 12px 16px;
    font-size: 13px;
    line-height: 1.65;
    margin: 0 0 8px 0;
    box-shadow: 0 10px 28px -18px rgba(13,27,51,0.28);
}
.finding-box-navy strong {
    color: var(--navy);
}

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
   테이블 / 탭 / 폼
───────────────────────────── */
div[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid rgba(13, 27, 51, 0.08);
    box-shadow: var(--shadow-card);
}

.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 2px solid rgba(13,27,51,0.10);
    padding-bottom: 0;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 7px 20px;
    background: rgba(255,255,255,0.50);
    border: 1px solid rgba(13,27,51,0.10);
    border-bottom: none;
    font-weight: 600;
    font-size: 13px;
    letter-spacing: 0.01em;
    color: #64748B;
}

.stTabs [aria-selected="true"] {
    background: #003278 !important;
    color: white !important;
    border-color: #003278 !important;
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
   페이지 컴포넌트
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
    margin-bottom: 20px;
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

div[data-testid="stVerticalBlockBorderWrapper"] {
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
    padding: 22px !important;
    margin: 12px 0 24px 0 !important;
    transition: all 0.18s ease !important;
}

div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-1px);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.90) inset,
        0 14px 32px -14px rgba(13,27,51,0.28),
        0 4px 10px -3px rgba(13,27,51,0.12) !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"] {
    gap: 0.72rem;
}

/* ─────────────────────────────
   영상 모드 선택기
───────────────────────────── */
.video-mode-head {
    width: 100%;
    box-sizing: border-box;
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(13, 27, 51, 0.08) 0%, transparent 54%),
        linear-gradient(145deg, rgba(247, 250, 255, 0.98), rgba(235, 242, 253, 0.96));
    border: 1px solid rgba(13, 27, 51, 0.18);
    border-bottom: none;
    border-radius: 16px 16px 0 0;
    padding: 13px 16px 7px 16px;
    margin: 12px 0 0 0;
    box-shadow:
        0 10px 26px -18px rgba(13, 27, 51, 0.30),
        0 2px 6px -3px rgba(13, 27, 51, 0.12);
}

.video-mode-head strong {
    color: #0D1B33;
    font-size: 13px;
    font-weight: 850;
    margin-right: 10px;
}

.video-mode-head span {
    color: #667085;
    font-size: 12px;
    font-weight: 500;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"],
.element-container:has(.video-mode-head) + .element-container {
    width: 100% !important;
    max-width: 100% !important;
    display: block !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[data-testid="stRadio"],
.element-container:has(.video-mode-head) + .element-container div[data-testid="stRadio"] {
    width: 100% !important;
    max-width: 100% !important;
    display: block !important;
    box-sizing: border-box !important;
    background:
        linear-gradient(145deg, rgba(247, 250, 255, 0.98), rgba(235, 242, 253, 0.96));
    border: 1px solid rgba(13, 27, 51, 0.18);
    border-top: none !important;
    border-radius: 0 0 16px 16px;
    padding: 4px 16px 14px 16px !important;
    margin-top: -1px !important;
    margin-bottom: 14px !important;
    overflow: visible !important;
    box-shadow:
        0 10px 26px -18px rgba(13, 27, 51, 0.30),
        0 2px 6px -3px rgba(13, 27, 51, 0.12);
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[data-testid="stRadio"] > div,
.element-container:has(.video-mode-head) + .element-container div[data-testid="stRadio"] > div {
    width: 100% !important;
    max-width: 100% !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"],
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 8px !important;
    align-items: center !important;
    flex-wrap: wrap !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    column-gap: 0 !important;
    background: rgba(13, 27, 51, 0.07);
    border: 1px solid rgba(13, 27, 51, 0.16);
    border-radius: 999px;
    height: 34px !important;
    padding: 0 16px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    cursor: pointer;
    transition: all 0.16s ease;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label > div:first-child,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label > div:first-child {
    display: none !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label > div:last-child,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label > div:last-child,
div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label [data-testid="stMarkdownContainer"],
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    height: 100% !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label > div:last-child > div,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label > div:last-child > div {
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    align-items: center !important;
    height: 100% !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label p,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label p {
    color: #0D1B33 !important;
    font-size: 12px !important;
    font-weight: 750 !important;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1 !important;
    transform: translateY(-2px) !important;
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label:hover,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label:hover {
    background: rgba(13, 27, 51, 0.12);
    border-color: rgba(13, 27, 51, 0.26);
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label:has(input:checked),
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label:has(input:checked) {
    background: #0D1B33 !important;
    border-color: #0D1B33 !important;
    box-shadow: 0 8px 18px -12px rgba(13, 27, 51, 0.55);
}

div[data-testid="stElementContainer"]:has(.video-mode-head) + div[data-testid="stElementContainer"] div[role="radiogroup"] label:has(input:checked) p,
.element-container:has(.video-mode-head) + .element-container div[role="radiogroup"] label:has(input:checked) p {
    color: #FFFFFF !important;
}

/* ─────────────────────────────
   영상 제목 아이콘
───────────────────────────── */
.video-case-title {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 8px;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 16px;
    font-weight: 800;
    color: #0D1B33;
    letter-spacing: -0.015em;
}

.video-case-title .case-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 18px;
    height: 18px;
    font-size: 17px;
    line-height: 1;
    color: #0D1B33;
}

.video-case-title.success .case-icon,
.video-case-title.fail .case-icon {
    color: #0D1B33;
    background: transparent;
    border: none;
}

.video-case-title .case-n {
    color: #667085;
    font-size: 13px;
    font-weight: 500;
}

/* ─────────────────────────────
   사이드바 커스텀 내비게이션 버튼
───────────────────────────── */
section[data-testid="stSidebar"] .stButton {
    margin: 1px 8px !important;
    padding: 0 !important;
}
section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    background: transparent !important;
    color: rgba(255,255,255,0.85) !important;
    border: none !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    padding: 10px 14px !important;
    font-family: "Manrope", sans-serif !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    margin: 0 !important;
    transition: background 0.15s, border-color 0.15s !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: #10264A !important;
    color: #FFFFFF !important;
}
section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    background: rgba(26,50,87,0.90) !important;
    color: #FFFFFF !important;
    border-left: 3px solid #B31922 !important;
    font-weight: 700 !important;
}
section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:hover {
    background: #1E3A65 !important;
}
section[data-testid="stSidebar"] .stButton > button p {
    font-family: "bootstrap-icons", "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    margin: 0 !important;
    text-align: left !important;
    letter-spacing: 0 !important;
    word-spacing: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    margin: 4px 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details,
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
    background: rgba(13,27,51,0.40) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details {
    border-radius: 10px !important;
    overflow: hidden !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary {
    list-style: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary::-webkit-details-marker {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"],
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"] p {
    color: rgba(255,255,255,0.88) !important;
    font-family: "Manrope", sans-serif !important;
    font-size: 14px !important;
    font-weight: 800 !important;
    padding: 10px 14px !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"] p {
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary:hover,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"]:hover {
    background: #10264A !important;
}
section[data-testid="stSidebar"] [data-testid="stExpander"] details > summary svg,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"] svg {
    fill: rgba(255,255,255,0.60) !important;
    stroke: rgba(255,255,255,0.60) !important;
    width: 16px !important;
    height: 16px !important;
    flex: 0 0 16px !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
    gap: 7px !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    align-items: flex-start !important;
    width: 100% !important;
    min-height: 48px !important;
    padding: 10px 12px !important;
    margin: 0 !important;
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.16) !important;
    border-left: 3px solid transparent !important;
    border-radius: 10px !important;
    transition: background 0.15s, border-color 0.15s, color 0.15s !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.14) !important;
    border-color: rgba(255,255,255,0.28) !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
    background: #FFFFFF !important;
    border-color: #FFFFFF !important;
    border-left-color: #B31922 !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label p {
    color: rgba(255,255,255,0.92) !important;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    font-size: 13px !important;
    font-weight: 650 !important;
    line-height: 1.35 !important;
    white-space: normal !important;
    word-break: keep-all !important;
}
section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p {
    color: #0D1B33 !important;
    font-weight: 800 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: rgba(255,255,255,0.72) !important;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    line-height: 1.5 !important;
}
.agent-sidebar-list {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin: 8px 8px 2px 8px;
}
.agent-sidebar-row {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 8px 10px;
    border-radius: 8px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.92);
    font-family: "Manrope", "Noto Sans KR", sans-serif;
    font-size: 12.5px;
    font-weight: 650;
    line-height: 1.35;
    word-break: keep-all;
}
.agent-sidebar-row.locked {
    color: rgba(255,255,255,0.68);
    background: rgba(255,255,255,0.045);
}
.agent-sidebar-icon {
    flex: 0 0 auto;
    width: 15px;
    color: #FFFFFF;
    font-family: "bootstrap-icons";
    font-size: 13px;
    line-height: 1.35;
    text-align: center;
}
.agent-sidebar-row.locked .agent-sidebar-icon {
    color: rgba(255,255,255,0.56);
}
.agent-sidebar-code {
    color: #FFFFFF;
    font-family: "JetBrains Mono", monospace;
    font-size: 11.5px;
    font-weight: 700;
}
.agent-sidebar-desc {
    color: rgba(255,255,255,0.78);
    font-weight: 600;
}
.agent-sidebar-note {
    margin: 8px 8px 2px 8px;
    padding: 8px 10px;
    border-radius: 8px;
    background: rgba(179,25,34,0.16);
    border: 1px solid rgba(179,25,34,0.28);
    color: #FFFFFF;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif;
    font-size: 12.5px;
    font-weight: 700;
    line-height: 1.4;
}
.sidebar-nav-section {
    display: block;
    color: rgba(255,255,255,0.48);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 12px 18px 5px 18px;
}

section[data-testid="stSidebar"] .stButton > button {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    text-align: left !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    width: 100% !important;
    text-align: left !important;
    justify-content: flex-start !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    white-space: nowrap !important;
}


/* ─────────────────────────────────────────────────────────
   다크 모드 - OS prefers-color-scheme CSS 변수 오버라이드
───────────────────────────────────────────────────────── */
</style>
""", unsafe_allow_html=True)


# ── 페이지 라우팅 상태 ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "overview"

_PITCHER_PAGES = {"leiter", "webb", "garcia", "armstrong", "jackson"}


def _nav_btn(label, page_id):
    is_active = st.session_state["page"] == page_id
    if st.button(label, key=f"_nb_{page_id}", use_container_width=True,
                 type="primary" if is_active else "secondary"):
        st.session_state["page"] = page_id
        st.rerun()


# ── 사이드바 ────────────────────────────────────────────────────────
logo_path = ASSETS / "images" / "logo.png"

with st.sidebar:
    if logo_path.exists():
        logo_base64 = image_to_base64(logo_path)
        brand_html = f'<img class="sidebar-logo-full" src="data:image/png;base64,{logo_base64}">'
    else:
        brand_html = """
        <div style="font-family:Sora, Manrope, sans-serif; color:#0D1B33; font-size:18px; font-weight:800; line-height:1.05;">
            Monday<br>Likes Baseball
        </div>
        """

    st.markdown(f"""
    <div class="sidebar-sticky-head">
    <div class="sidebar-brand-shell">
        {brand_html}
    </div>
    <div class="sidebar-project-wrap">
        <div class="sidebar-section-label">CLIENT PROJECT</div>
        <div class="sidebar-project-title">Texas Rangers</div>
        <div class="sidebar-project-sub">2025 Residual Analysis</div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-note-wrap">
        <div class="sidebar-section-label">PROJECT NOTE</div>
        <div class="sidebar-note-body">
            <span class="note-line">TEX 2025 잔차 -9.06승 원인 진단</span>
            <span class="note-subline">— 수동 시나리오 · Grid/Pareto 후보 비교</span>
        </div>
    </div>
    <div class="sidebar-divider"></div>
    """, unsafe_allow_html=True)

    _nav_btn(" Overview",    "overview")
    _nav_btn(" Simulation",  "simulation")
    _nav_btn(" Comparison",  "comparison")
    _nav_btn(" AI Agent", "ai_agent")
    _nav_btn(" Conclusions", "conclusions")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-nav-section">Pitcher Analysis</span>', unsafe_allow_html=True)
    _cur = st.session_state["page"]
    with st.expander("Roster", expanded=(_cur in _PITCHER_PAGES)):
        _nav_btn(" Leiter",    "leiter")
        _nav_btn(" Webb",      "webb")
        _nav_btn(" Garcia",    "garcia")
        _nav_btn(" Armstrong", "armstrong")
        _nav_btn(" Jackson",   "jackson")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)

    _missing_v5_outputs = [name for name, path in _V5_OUTPUT_FILES.items() if not path.exists()]
    if _missing_v5_outputs:
        st.warning("v5 출력 파일 누락: " + ", ".join(_missing_v5_outputs))

    _nav_btn(" Methodology", "methodology")


# ── 페이지 라우팅 ──────────────────────────────────────────────────
page = st.session_state.get("page", "overview")

if page == "overview":
    v_overview.show()
elif page == "simulation":
    v_simulation.show()
elif page == "methodology":
    v_methodology.show()
elif page == "leiter":
    v_leiter.show()
elif page == "webb":
    v_webb.show()
elif page == "garcia":
    v_garcia.show()
elif page == "armstrong":
    v_armstrong.show()
elif page == "jackson":
    v_jackson.show()
elif page == "comparison":
    v_comparison.show()
elif page == "ai_agent":
    v_ai_agent.show()
elif page == "conclusions":
    v_conclusions.show()

st.caption("© 2026. M.L.B Co. All rights reserved.")
