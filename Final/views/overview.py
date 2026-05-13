import streamlit as st
from shared import data, kpi_card, page_hero, glossary_box, BASEBALL_TERMS

_HR = "<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>"


def show():
    page_hero(
        "Overview",
        "2025 TEX 잔차 분석 대시보드",
        "이 대시보드는 2025 텍사스 레인저스가 통계적으로 기대되는 승수(90.1승)보다 9승이나 적은 81승에 그친 이유를 분석합니다.<br>"
        "경기 운영·선수 상태·투구 동작을 단계적으로 살펴보고, 마지막으로 어떤 변화가 승수를 개선할 수 있는지 비교합니다.",
        [("Pythagorean Residual", "white"), ("Game & Player Analysis", "white"), ("Scenario Simulation", "white")],
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        kpi_card("기대 대비 부진 규모", "-9.1승", "실제 81승 vs 통계 기대치 90.1승", accent="red")
    with col2:
        kpi_card("분석 흐름", "3단계", "경기 운영 → 투구 동작 → 시뮬레이션", accent="navy")
    with col3:
        kpi_card("하이 레버리지 부진 선수", "5명", "접전·세이브 상황에서 부진한 투수", accent="red")
    with col4:
        kpi_card("AI Agent", "대화형 분석", "시나리오·팀 비교·개선안 질의 가능", accent="navy")

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 1. 분석 배경")

    st.markdown("""
    야구에는 득점과 실점을 바탕으로 "이 팀이라면 몇 승을 해야 했는가"를 계산하는 공식이 있습니다(피타고리안 기대 승수).  
    **2025 텍사스 레인저스는 이 기대치보다 9.1승 적게 이겼습니다.** 이 9.1승 차이가 이 분석의 출발점입니다.  
    그 원인을 찾기 위해 먼저 경기 운영과 선수 성적으로 팀 전체 맥락을 확인하고,
    특히 하이 레버리지(세이브·1점차 접전)에 부진했던 투수 5명의 투구 동작을 3D 분석으로 비교합니다.
    """)

    st.markdown("""
    <div class="question-box" style="font-size:14.5px; line-height:1.78; padding:20px 24px; max-width:none; width:100%; margin-top:24px; border-left-width:6px; box-shadow:0 14px 30px -22px rgba(13,27,51,0.35);">
        <strong>핵심 질문:</strong><br>
        81승 팀이 왜 90.1승 기대치에 미치지 못했는가?<br>
        그 차이는 경기 운영, 선수 상태, 투구 동작, 개선 시나리오 비교에서 어떻게 설명되는가?
    </div>
    """, unsafe_allow_html=True)

    st.markdown(_HR, unsafe_allow_html=True)

    st.markdown("## 2. 야구 지표 용어")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:18px;">
            <div class="section-copy">표와 그래프를 읽기 전에 필요한 용어만 짧게 정리했습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    glossary_box("야구 지표 용어", {
        "BB/9": BASEBALL_TERMS["BB/9"],
        "ERA": BASEBALL_TERMS["ERA"],
        "FIP": BASEBALL_TERMS["FIP"],
        "WAR": BASEBALL_TERMS["WAR"],
        "Clutch": BASEBALL_TERMS["Clutch"],
        "세이브(SV) / 블론 세이브(BS)": BASEBALL_TERMS["세이브(SV) / 블론 세이브(BS)"],
    }, mb=0, show_header=False)

    st.markdown(_HR, unsafe_allow_html=True)

    st.markdown("## 3. 분석 대상 투수 5명")
    st.markdown("""
    <div class="glass-card" style="margin-bottom:20px;">
        <div class="chart-caption">
            하이 레버리지에서 부진했던 투수 4명과 비교 기준이 된 Armstrong, 총 5명을 선정했습니다.<br>
            <span style="display:inline-block; margin-top:6px; line-height:1.9;">
                · 선발: 중요한 경기에 자주 등판한 횟수 기준<br>
                · 불펜: 세이브 기회 5회 이상 등판 기준
            </span>
        </div>
        <div style="margin-top:8px; font-size:11.5px; color:#94A3B8; line-height:1.6;">
            * 하이 레버리지(결정적 장면): 한 번의 실수가 승패를 바꿀 수 있는 긴장된 상황 — 접전 후반, 세이브, 블론 세이브 등
        </div>
    </div>
    """, unsafe_allow_html=True)

    _PITCHER_CARDS = [
        {
            "name": "Webb",
            "type": "선발",
            "type_color": "#003278",
            "bg": "rgba(240,245,255,0.97)",
            "border": "rgba(0,50,120,0.22)",
            "finding": "ERA 3.00 vs FIP 4.30",
            "action": "방어율은 좋지만 실제 실력과 차이 가능성",
        },
        {
            "name": "Leiter",
            "type": "선발",
            "type_color": "#003278",
            "bg": "rgba(240,245,255,0.97)",
            "border": "rgba(0,50,120,0.22)",
            "finding": "볼넷 허용이 지속적으로 많음",
            "action": "위기 상황에서 제구가 무너지는 경향",
        },
        {
            "name": "Garcia",
            "type": "마무리",
            "type_color": "#9A3412",
            "bg": "rgba(255,247,237,0.97)",
            "border": "rgba(154,52,18,0.22)",
            "finding": "하이 레버리지 성적 팀 최하위",
            "action": "일반 상황은 괜찮지만<br>중요한 순간에 가장 약함",
        },
        {
            "name": "Armstrong",
            "type": "비교 기준",
            "type_color": "#166534",
            "bg": "rgba(240,253,244,0.97)",
            "border": "rgba(22,101,52,0.22)",
            "finding": "ERA 2.31 / 하이 레버리지 성적 양호",
            "action": "불펜에서 가장 안정적이었던 투수",
        },
        {
            "name": "Jackson",
            "type": "불펜",
            "type_color": "#4A5568",
            "bg": "rgba(247,250,252,0.97)",
            "border": "rgba(74,85,104,0.22)",
            "finding": "WAR 마이너스 (유일)",
            "action": "불펜 중 유일하게<br>팀 전력에 손해가 된 투수",
        },
    ]

    case_cols = st.columns(5, gap="small")
    for col, card in zip(case_cols, _PITCHER_CARDS):
        with col:
            st.markdown(f"""
            <div style="background:{card['bg']};border:1px solid {card['border']};border-radius:14px;
                        padding:18px 12px 16px;text-align:center;min-height:214px;">
                <div style="display:inline-block;background:{card['type_color']};color:#fff;
                            font-size:11.5px;font-weight:800;padding:3px 11px;border-radius:20px;
                            margin-bottom:10px;letter-spacing:0.4px;white-space:nowrap;">
                    {card['type']}
                </div>
                <div style="font-size:20px;font-weight:800;color:#0D1B33;margin-bottom:6px;">
                    {card['name']}
                </div>
                <div style="font-size:13.5px;font-weight:800;color:{card['type_color']};margin-bottom:8px;line-height:1.45;">
                    {card['finding']}
                </div>
                <div style="font-size:13px;color:#475569;line-height:1.55;">
                    → {card['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 4. 분석 흐름")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="card">
            <span class="pill pill-card-red">1. Residual Diagnosis</span>
            <h3>경기 운영 분석</h3>
            <p><b>득점과 승패가 어긋난 구간 확인</b></p>
            <ul>
                <li>1점차, 연장전, 세이브 상황</li>
                <li>월별 득실과 일정 흐름</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 9승 차이가 어디서 발생했는지 확인</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <span class="pill pill-card-navy">2. Motion Evidence</span>
            <h3>투구 동작 분석</h3>
            <p><b>투구 폼 차이와 외부 요인 분리</b></p>
            <ul>
                <li>Webb·Garcia 등 선수별 케이스</li>
                <li>통계적 차이 크기와 유의성 비교</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 코칭으로 고칠 수 있는지 판단</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="card">
            <span class="pill pill-gray">3. Decision Simulation</span>
            <h3>개선 시나리오 비교</h3>
            <p><b>다양한 개선안을 같은 기준으로 정리</b></p>
            <ul>
                <li>승수 개선 폭과 예측 안정성 비교</li>
                <li>공격적·균형·보수적 선택지 확인</li>
            </ul>
            <p style="color:#64748B; font-size:13px;">→ 우선 검토할 운영 조합 선정</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 5. 대시보드 사용법")
    st.markdown("""
    좌측 사이드바에서 페이지를 이동할 수 있습니다.

    - **Simulation**: 기준 시즌 확인, 다양한 개선 시나리오 비교
    - **Interactive Simulation**: 타자·투수 조건을 직접 조정하며 경기 결과 변화 확인
    - **Comparison**: 투수 5명의 투구 동작 차이를 한눈에 비교
    - **각 투수**: 선수별 세부 투구 동작 분석
    - **Methodology**: AI 동작 분석 모델 선택 근거와 지표 설명
    - **Conclusions**: 분석 결론 요약, 운영 권고, PDF 보고서 출력
    - **AI Agent**: 시나리오·팀 비교·개선안 관련 질문 가능
    """)
