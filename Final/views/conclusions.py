import streamlit as st
import pandas as pd
from shared import (
    page_hero, finding_box, section_header, kpi_card,
    REPORT_FINDINGS, build_player_report_pdf, build_team_report_pdf,
)


def show():
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
                삼진 시 상하체 분리와 kinetic chain이 안정적이고, 볼넷 시에는 동시 회전에 가까워지며 정확성이 흔들리는 흐름입니다.
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
        try:
            _player_pdf = build_player_report_pdf(selected_report_player)
            st.download_button(
                "선수별 보고서 PDF 출력",
                data=_player_pdf,
                file_name=f"tex_2025_player_report_{selected_report_player.lower()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _e:
            st.error(f"PDF 생성 오류: {_e}")
    with report_col_2:
        st.markdown("""
        <div class="glass-card">
            <b>팀 요약 보고서</b><br>
            잔차 분석 목적, 분석 흐름, 대표 투수별 핵심 해석, 최종 결론을 한 문서로 정리합니다.
        </div>
        """, unsafe_allow_html=True)
        try:
            _team_pdf = build_team_report_pdf()
            st.download_button(
                "팀 요약 보고서 PDF 출력",
                data=_team_pdf,
                file_name="tex_2025_team_residual_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _e:
            st.error(f"PDF 생성 오류: {_e}")