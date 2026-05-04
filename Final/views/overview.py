import streamlit as st
from shared import data, kpi_card, section_badges


def show():
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
        결과 분기 상황(삼진/볼넷, 세이브/블론 세이브)을 3D 키네마틱 지표로 검증합니다.\
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