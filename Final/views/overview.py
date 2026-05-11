import streamlit as st
from shared import data, kpi_card, section_badges, glossary_box, BASEBALL_TERMS


def show():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">Pythagorean Residual</span>
        <span class="pill pill-white">Game & Player Analysis</span>
        <span class="pill pill-white">Scenario Simulation</span>
        <h1>2025 TEX 잔차 분석 대시보드</h1>
        <p>
        이 대시보드의 목표는 2025 텍사스 레인저스가 실제 승수보다 피타고리안 기대 승수에서 9.06승 낮게 끝난 이유를 설명하는 것입니다.<br>
        경기력·선수·모션 근거로 원인을 좁히고, Simulation에서는 수동 시나리오와 Grid/Pareto 후보를 같은 기준으로 비교합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("잔차 규모", "-9.06승", "실제 81승 vs 피타고리안 기대 90.06승", accent="red")
    with col2:
        kpi_card("분석 흐름", "3단계", "경기력 분석 → 모션 분석 → 시뮬레이션", accent="navy")
    with col3:
        kpi_card("하이 레버리지 부진 케이스", "5명", "접전·결정적 장면 투수 세부 사례", accent="red")
    with col4:
        kpi_card("AI Agent", "대화형 분석", "시나리오·팀 비교·최적화 질의 지원", accent="navy")

    st.markdown("<hr style='margin: 36px 0 36px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("## 분석 배경")

    st.markdown("""
    **2025 텍사스 레인저스의 Pythagorean 잔차 -9.06승** — 팀이 기대 승수보다 9승가량 적게 기록한 격차가 출발점입니다.
    따라서 본 프로젝트의 중심 질문은 "투수 폼이 달랐는가"가 아니라 **왜 실제 승수와 기대 승수 사이에 큰 잔차가 발생했는가**입니다.

    이를 위해 경기력 분석과 선수 분석으로 팀 단위 맥락을 먼저 확인한 뒤,
    잔차와 연결될 수 있는 투수 운영 케이스를 더 깊게 보기 위해 하이 레버리지 상황에서 부진했던 투수 5명의
    결과 분기 상황(삼진/볼넷, 세이브/블론 세이브)을 3D 키네마틱 지표로 검증합니다.
    """)

    st.markdown("""
    <div class="question-box" style="font-size:14.5px; line-height:1.78; padding:20px 24px; max-width:none; width:100%; margin-top:24px; border-left-width:6px; box-shadow:0 14px 30px -22px rgba(13,27,51,0.35);">
        <strong>핵심 질문:</strong><br>
        81승 팀이 왜 90.06승 기대치에 미치지 못했는가?<br>
        그 차이는 경기 운영, 선수 상태, 투수 모션, 의사결정 후보 비교에서 어떻게 설명되는가?
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)

    # ── 하이 레버리지 부진 케이스 5명 카드 ──────────────────────────────
    glossary_box("야구 지표 용어", {
        "BB/9": BASEBALL_TERMS["BB/9"],
        "ERA": BASEBALL_TERMS["ERA"],
        "FIP": BASEBALL_TERMS["FIP"],
        "WAR": BASEBALL_TERMS["WAR"],
        "세이브(SV) / 블론 세이브(BS)": BASEBALL_TERMS["세이브(SV) / 블론 세이브(BS)"],
    })

    st.markdown("""
    <div class="glass-card" style="margin-bottom:20px;">
        <div class="chart-title">분석 대상 투수 5명</div>
        <div class="chart-caption">
            하이 레버리지 상황에서 부진했던 투수 4명 + 비교 기준 Armstrong, 총 5명을 선정했습니다.<br>
            <span style="display:inline-block; margin-top:6px; line-height:1.9;">
                · 선발: 하이 레버리지 등판 횟수 기준<br>
                · 불펜: 세이브 기회 5회 이상 등판 기준
            </span>
        </div>
        <div style="margin-top:8px; font-size:11.5px; color:#94A3B8; line-height:1.6;">
            * 하이 레버리지: 득점 확률 변화가 큰 접전 후반·세이브·블론 세이브 같은 긴장된 상황
        </div>
    </div>
    """, unsafe_allow_html=True)

    _CASE_INFO = {
        "Leiter":    {"role": "선발", "sit": "삼진 vs 볼넷", "stat": "BB/9 만성 제구 이슈",       "reason": "위기 상황에서 볼넷이 잦음",                  "tone": "navy"},
        "Webb":      {"role": "선발", "sit": "삼진 vs 볼넷", "stat": "ERA 3.00 vs FIP 4.30",    "reason": "ERA 대비 실제 실력이 낮을 수 있음",          "tone": "navy"},
        "Garcia":    {"role": "불펜", "sit": "세이브 vs 블론 세이브","stat": "Clutch −1.164 (팀 최하위)","reason": "수치는 좋지만 결정적 순간에 가장 약함",       "tone": "red"},
        "Armstrong": {"role": "불펜", "sit": "세이브 vs 블론 세이브","stat": "ERA 2.31 / Clutch 0.867", "reason": "불펜에서 가장 꾸준했던 투수 (비교 기준)",     "tone": "green"},
        "Jackson":   {"role": "불펜", "sit": "세이브 vs 블론 세이브","stat": "WAR −0.33 (유일 음수)",    "reason": "팀에서 유일하게 마이너스 기여를 한 투수",     "tone": "red"},
    }
    _TONE = {
        "navy":  {"bg": "rgba(240,245,255,0.97)", "border": "rgba(0,50,120,0.22)",   "badge": "#003278", "stat": "#003278"},
        "red":   {"bg": "rgba(255,245,245,0.97)", "border": "rgba(179,25,34,0.22)",  "badge": "#B3191A", "stat": "#B3191A"},
        "green": {"bg": "rgba(240,253,244,0.97)", "border": "rgba(22,101,52,0.22)",  "badge": "#166534", "stat": "#166534"},
    }

    case_cols = st.columns(5, gap="small")
    for col, (name, info) in zip(case_cols, _CASE_INFO.items()):
        meta = data["meta"]["pitchers"].get(name, {})
        n_a, n_b = meta.get("n_a", "—"), meta.get("n_b", "—")
        t = _TONE[info["tone"]]
        with col:
            st.markdown(f"""
            <div style="background:{t['bg']};border:1px solid {t['border']};border-radius:14px;
                        padding:18px 12px 16px;text-align:center;min-height:214px;">
                <div style="display:inline-block;background:{t['badge']};color:#fff;
                            font-size:11.5px;font-weight:800;padding:3px 11px;border-radius:20px;
                            margin-bottom:10px;letter-spacing:0.4px;">{info['role']}</div>
                <div style="font-size:20px;font-weight:800;color:#0D1B33;margin-bottom:6px;">{name}</div>
                <div style="font-size:11.5px;color:#667085;margin-bottom:11px;line-height:1.45;">{info['sit']}<br>(n={n_a}/{n_b})</div>
                <div style="font-size:12.5px;font-weight:800;color:{t['stat']};margin-bottom:8px;line-height:1.35;">{info['stat']}</div>
                <div style="font-size:12px;color:#475569;line-height:1.55;">{info['reason']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown("<hr style='margin: 36px 0 36px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("## 분석 흐름")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <span class="pill pill-card-red">1. Residual Diagnosis</span>
            <h3>경기력 분석</h3>
            <p><b>득실과 승패가 어긋난 구간 확인</b></p>
            <ul>
                <li>1점차, 연장, 세이브 상황</li>
                <li>월별 득실·스케줄 맥락</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ -9.06승 잔차의 발생 지점 확인</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <span class="pill pill-card-navy">2. Motion Evidence</span>
            <h3>하이 레버리지 부진 케이스 모션 분석</h3>
            <p><b>폼 교정 가능성과 외부 요인 분리</b></p>
            <ul>
                <li>Webb·Garcia 등 선수별 케이스</li>
                <li>Cohen's d와 p-value 비교</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 코칭/운영 판단의 세부 근거 확인</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <span class="pill pill-gray">3. Decision Simulation</span>
            <h3>의사결정 후보 비교</h3>
            <p><b>수동·Grid·Pareto 후보를 같은 표로 정리</b></p>
            <ul>
                <li>승수 개선 폭과 예측 안정성 비교</li>
                <li>공격적·균형·보수 선택지 확인</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 우선 검토할 운영 조합 선정</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='margin: 36px 0 36px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("## 대시보드 사용법")
    st.markdown("""
    좌측 사이드바에서 페이지 이동:

    - **Simulation**: 기준선 확인, 수동 시나리오와 Grid/Pareto 후보 비교
    - **Comparison**: 분석 대상 투수 5명의 모션 근거 통합 비교
    - **각 투수**: 잔차 원인 세부화를 위한 개별 투수 모션 분석
    - **AI Agent**: 시나리오·최적화·팀 비교 질의 지원
    - **Conclusions**: 잔차 해석, 운영 권고, PDF 보고서 출력
    - **Methodology**: MotionAGFormer 채택 근거와 측정 지표 정의
    """)
