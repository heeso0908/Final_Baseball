"""TEX 2025 피타고리안 Residual Analysis Dashboard."""
# ── 한글 폰트 설정 ── Streamlit 명령어보다 먼저 실행해야 함 ──
import platform as _platform
import sys as _sys
import glob as _glob
from pathlib import Path

# ── 부모 디렉토리(Final/) sys.path 추가 ── simulator, integrated_sim 등 분석 라이브러리 import용 ──
_PARENT = Path(__file__).resolve().parent.parent
if str(_PARENT) not in _sys.path:
    _sys.path.insert(0, str(_PARENT))

import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
from PIL import Image as _Image


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

_PAGE_ICON_PATH = Path(__file__).resolve().parent / "assets" / "images" / "logo_page.png"
try:
    _PAGE_ICON = _Image.open(_PAGE_ICON_PATH) if _PAGE_ICON_PATH.exists() else "⚾"
except Exception:
    _PAGE_ICON = "⚾"

st.set_page_config(
    page_title="TEX 2025 Residual Analysis",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── 공유 유틸리티 및 뷰 모듈 임포트 (set_page_config 이후) ─────────
from shared import ASSETS, image_to_base64
import views.overview as v_overview
import views.simulation as v_simulation
import views.interactive_sim as v_interactive_sim
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

[data-testid="stSelectbox"] label,
[data-testid="stSlider"] label {
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #1E293B !important;
    letter-spacing: -0.01em;
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
.sidebar-brand-link {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100%;
    text-decoration: none !important;
    cursor: pointer;
}

.sidebar-project-wrap {
    padding: 22px 18px 18px 18px;
    margin-bottom: 10px;
    position: relative;
}
.sidebar-project-wrap::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 8px;
    right: 8px;
    height: 1px;
    background: rgba(255,255,255,0.08);
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
    margin: 12px 8px 14px 8px;
}

.sidebar-note-wrap {
    padding: 12px 18px 12px 18px;
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

.glass-card .chart-title,
.glass-card.glass-card-accent .chart-title,
.glass-card.glass-card-red .chart-title,
.glass-card.glass-card-navy .chart-title,
.glass-card.glass-card-amber .chart-title {
    font-size: 18px;
}

.glass-card .chart-caption {
    font-size: 15px;
    line-height: 1.65;
    color: #334155;
    margin-top: 10px;
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
    font-size: 15px;
    line-height: 1.7;
    margin: 15px 0;
    box-shadow: 0 10px 28px -18px rgba(179,25,34,0.28);
}
.finding-box strong {
    display: block;
    color: var(--rangers-red);
    font-size: 17px;
    font-family: "Sora", "Manrope", sans-serif;
    letter-spacing: -0.015em;
    margin-bottom: 10px;
}
.finding-box-body {
    font-size: 15px;
    line-height: 1.7;
}

.finding-box-navy {
    background:
        radial-gradient(120% 80% at 100% 0%, rgba(13,27,51,0.09) 0%, transparent 55%),
        linear-gradient(145deg, rgba(255,255,255,0.90), rgba(255,255,255,0.72));
    border: 1px solid rgba(13,27,51,0.16);
    color: #344054;
    border-radius: 18px;
    padding: 12px 16px;
    font-size: 15px;
    line-height: 1.7;
    margin: 0 0 8px 0;
    box-shadow: 0 10px 28px -18px rgba(13,27,51,0.28);
}
.finding-box-navy strong {
    color: var(--navy);
    font-size: 17px;
    font-family: "Sora", "Manrope", sans-serif;
    letter-spacing: -0.015em;
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
div[data-testid="stDataFrame"] th {
    color: #1B2435 !important;
    font-weight: 800 !important;
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
button[data-testid="baseButton-primary"] {
    background-color: #B31922 !important;
    border-color: #B31922 !important;
    color: #ffffff !important;
}
button[data-testid="baseButton-primary"]:hover {
    background-color: #9A1419 !important;
    border-color: #9A1419 !important;
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
.quote-card strong {
    display: block;
    font-size: 18px;
    font-family: "Sora", "Manrope", sans-serif;
    letter-spacing: -0.015em;
    margin-bottom: 10px;
}
.quote-card p {
    font-size: 15px;
    line-height: 1.7;
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
   - 기존 디자인 유지
   - Bootstrap Icons 유지
   - hover/active 박스 잘림 방지
   - AI Agent 선택 시 추가 패널 지원
───────────────────────────── */
section[data-testid="stSidebar"] .stButton {
    width: calc(100% - 20px) !important;
    max-width: calc(100% - 20px) !important;
    margin: 0 10px 8px 10px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 42px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    background: transparent !important;
    color: rgba(255,255,255,0.85) !important;
    border: 1px solid transparent !important;
    border-left: none !important;
    border-radius: 12px !important;
    padding: 10px 13px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    transition: background 0.16s ease, border-color 0.16s ease, color 0.16s ease !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #FFFFFF !important;
    border-color: rgba(255,255,255,0.10) !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"] {
    background: rgba(26,50,87,0.92) !important;
    color: #FFFFFF !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-left: none !important;
    font-weight: 800 !important;
    box-shadow: 0 10px 26px -18px rgba(0,0,0,0.45) !important;
}

section[data-testid="stSidebar"] .stButton > button[data-testid="baseButton-primary"]:hover {
    background: #1E3A65 !important;
    border-color: rgba(255,255,255,0.18) !important;
}

section[data-testid="stSidebar"] .stButton > button div,
section[data-testid="stSidebar"] .stButton > button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] .stButton > button p {
    width: 100% !important;

    /* p 전체를 flex로 두면 ::first-letter 조정이 잘 안 먹을 수 있어서 block으로 변경 */
    display: block !important;

    /* 텍스트는 일반 폰트 기준으로 정렬 */
    font-family: "Manrope", "Pretendard", "Noto Sans KR", "bootstrap-icons", sans-serif !important;

    color: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;

    margin: 0 !important;
    padding: 0 !important;
    text-align: left !important;

    line-height: 1.2 !important;
    letter-spacing: 0 !important;
    word-spacing: 0 !important;
    white-space: nowrap !important;
}

/* 첫 글자, 즉 Bootstrap icon 문자만 따로 정렬 */
section[data-testid="stSidebar"] .stButton > button p::first-letter {
    font-family: "bootstrap-icons" !important;
    font-size: 15px !important;
    font-weight: normal !important;
    line-height: 1 !important;
    vertical-align: -0.08em !important;
}

.sidebar-nav-section {
    display: block;
    color: rgba(255,255,255,0.48);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    padding: 14px 18px 7px 18px;
}

.sidebar-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 16px 10px 16px 10px;
}

/* Roster expander - 오른쪽 잘림 방지 */
section[data-testid="stSidebar"] [data-testid="stExpander"] {
    width: calc(100% - 20px) !important;
    max-width: calc(100% - 20px) !important;
    margin: 4px 10px 10px 10px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] details {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    background: rgba(13,27,51,0.40) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] details[open] > summary,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus,
section[data-testid="stSidebar"] [data-testid="stExpander"] summary:focus-visible,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"],
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"]:focus,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"]:focus-visible {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;

    background: rgba(13,27,51,0.40) !important;
    color: rgba(255,255,255,0.88) !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Roster 내부 버튼 폭 보정 */
section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton {
    width: calc(100% - 12px) !important;
    max-width: calc(100% - 12px) !important;
    margin: 0 6px 7px 6px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 38px !important;
    padding: 9px 12px !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stExpander"] summary p,
section[data-testid="stSidebar"] [data-testid="stExpander"] [data-testid="stExpanderHeader"] p {
    color: rgba(255,255,255,0.88) !important;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    font-size: 16px !important;
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
}

section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: rgba(255,255,255,0.72) !important;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif !important;
    line-height: 1.5 !important;
}

.agent-sidebar-list {
    display: flex;
    flex-direction: column;
    gap: 9px;
    margin: 10px 10px 4px 10px;
}

.agent-sidebar-row {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    padding: 10px 11px;
    border-radius: 12px;
    background: rgba(255,255,255,0.075);
    border: 1px solid rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.92);
    font-family: "Manrope", "Noto Sans KR", sans-serif;
    font-size: 12.5px;
    font-weight: 650;
    line-height: 1.38;
    word-break: keep-all;
}

.agent-sidebar-row.locked {
    color: rgba(255,255,255,0.68);
    background: rgba(255,255,255,0.045);
}

.agent-sidebar-icon {
    flex: 0 0 auto;
    width: 16px;
    color: #FFFFFF;
    font-family: "bootstrap-icons";
    font-size: 13px;
    line-height: 1.38;
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
    margin: 10px 10px 4px 10px;
    padding: 10px 11px;
    border-radius: 12px;
    background: rgba(179,25,34,0.16);
    border: 1px solid rgba(179,25,34,0.28);
    color: #FFFFFF;
    font-family: "Manrope", "Pretendard", "Noto Sans KR", sans-serif;
    font-size: 12.5px;
    font-weight: 700;
    line-height: 1.4;
}

/* ─────────────────────────────
   AI Agent 추가 사이드바 패널
───────────────────────────── */
.ai-agent-panel {
    margin: 18px 10px 16px 10px;
    padding: 16px 14px 15px 14px;
    border-radius: 18px;
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(255,255,255,0.10) 0%, transparent 52%),
        linear-gradient(145deg, rgba(255,255,255,0.085), rgba(255,255,255,0.045));
    border: 1px solid rgba(255,255,255,0.14);
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.12) inset,
        0 14px 34px -22px rgba(0,0,0,0.45);
}

.ai-agent-kicker {
    color: rgba(255,255,255,0.54);
    font-size: 10px;
    font-weight: 900;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.ai-agent-title {
    color: #FFFFFF;
    font-family: "Sora", "Manrope", sans-serif;
    font-size: 16px;
    font-weight: 850;
    letter-spacing: -0.02em;
    margin-bottom: 5px;
}

.ai-agent-desc {
    color: rgba(255,255,255,0.66);
    font-size: 12px;
    line-height: 1.55;
    margin-bottom: 13px;
}

.ai-agent-mini-grid {
    display: flex;
    flex-direction: column;
    gap: 9px;
}

.ai-agent-mini-card {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    padding: 10px 10px;
    border-radius: 13px;
    background: rgba(7,18,37,0.42);
    border: 1px solid rgba(255,255,255,0.10);
}

.ai-agent-mini-icon {
    color: #FFFFFF;
    font-size: 14px;
    line-height: 1.35;
    opacity: 0.9;
}

.ai-agent-mini-text {
    color: rgba(255,255,255,0.90);
    font-size: 12px;
    font-weight: 750;
    line-height: 1.35;
}

.ai-agent-mini-sub {
    display: block;
    color: rgba(255,255,255,0.56);
    font-size: 11px;
    font-weight: 500;
    margin-top: 3px;
}



</style>
""", unsafe_allow_html=True)


# ── 페이지 라우팅 상태 ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state["page"] = "overview"

_PITCHER_PAGES = {"leiter", "webb", "garcia", "armstrong", "jackson"}

# ── query param을 page 상태에 반영 ──────────────────────────────────
VALID_PAGES = {
    "overview",
    "simulation",
    "interactive_sim",
    "methodology",
    "leiter",
    "webb",
    "garcia",
    "armstrong",
    "jackson",
    "comparison",
    "ai_agent",
    "conclusions",
}

query_page = st.query_params.get("page", None)

if isinstance(query_page, list):
    query_page = query_page[0]

if query_page in VALID_PAGES:
    st.session_state["page"] = query_page


def _nav_btn(label, page_id):
    is_active = st.session_state["page"] == page_id
    if st.button(label, key=f"_nb_{page_id}", use_container_width=True,
                 type="primary" if is_active else "secondary"):
        st.session_state["page"] = page_id
        st.query_params["page"] = page_id
        st.rerun()


# ── 사이드바 ────────────────────────────────────────────────────────
logo_path = Path(__file__).resolve().parent / "assets" / "images" / "logo.png"

with st.sidebar:
    if logo_path.exists():
        logo_base64 = image_to_base64(logo_path)
        brand_html = (
            f'<a class="sidebar-brand-link" href="?page=overview" target="_self" aria-label="Overview로 이동">'
            f'<img class="sidebar-logo-full" src="data:image/png;base64,{logo_base64}">'
            f'</a>'
        )
    else:
        brand_html = """
        <a class="sidebar-brand-link" href="?page=overview" target="_self" aria-label="Overview로 이동">
        <div style="font-family:Sora, Manrope, sans-serif; color:#0D1B33; font-size:18px; font-weight:800; line-height:1.05;">
            Monday<br>Likes Baseball
        </div>
        </a>
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

    st.markdown('<span class="sidebar-nav-section">Analysis</span>', unsafe_allow_html=True)
    _nav_btn("  Overview",    "overview")
    _nav_btn("  Simulation",  "simulation")
    _nav_btn("  Interactive Sim",  "interactive_sim")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-nav-section">Motion Analysis</span>', unsafe_allow_html=True)
    _nav_btn("  Comparison",  "comparison")

    _cur = st.session_state["page"]
    with st.expander("Roster", expanded=(_cur in _PITCHER_PAGES)):
        _nav_btn("  Webb",      "webb")
        _nav_btn("  Leiter",    "leiter")
        _nav_btn("  Garcia",    "garcia")
        _nav_btn("  Armstrong", "armstrong")
        _nav_btn("  Jackson",   "jackson")
    _nav_btn("  Methodology", "methodology")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-nav-section">Output</span>', unsafe_allow_html=True)
    _nav_btn("  Conclusions", "conclusions")

    st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.markdown('<span class="sidebar-nav-section">Assistant</span>', unsafe_allow_html=True)
    _nav_btn("  AI Agent", "ai_agent")

    # AI Agent 선택 시에만 추가 사이드바 패널 표시
    if st.session_state.get("page") == "ai_agent":
        st.markdown("""
        <div class="sidebar-divider"></div>
        <div class="ai-agent-panel">
            <div class="ai-agent-kicker">AI Agent Panel</div>
            <div class="ai-agent-title">TEX 분석 어시스턴트</div>
            <div class="ai-agent-desc">
                시나리오, 게임로그, 팀 비교, historical 데이터를 질의형으로 확인하는 전용 작업 영역입니다.
            </div>
            <div class="ai-agent-mini-grid">
                <div class="ai-agent-mini-card">
                    <i class="bi bi-stars ai-agent-mini-icon"></i>
                    <div class="ai-agent-mini-text">
                        Plan-based Tools
                        <span class="ai-agent-mini-sub">플랜에 따라 도구 접근 범위 차등</span>
                    </div>
                </div>
                <div class="ai-agent-mini-card">
                    <i class="bi bi-database-check ai-agent-mini-icon"></i>
                    <div class="ai-agent-mini-text">
                        CSV Grounded
                        <span class="ai-agent-mini-sub">업로드된 분석 데이터 기반 응답</span>
                    </div>
                </div>
                <div class="ai-agent-mini-card">
                    <i class="bi bi-chat-dots ai-agent-mini-icon"></i>
                    <div class="ai-agent-mini-text">
                        Scenario Q&A
                        <span class="ai-agent-mini-sub">승수 변화·잔차 원인 질의 지원</span>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── 페이지 라우팅 ──────────────────────────────────────────────────
page = st.session_state.get("page", "overview")

if page == "overview":
    v_overview.show()
elif page == "simulation":
    v_simulation.show()
elif page == "interactive_sim":
    v_interactive_sim.show()
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
