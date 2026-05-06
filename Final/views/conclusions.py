import streamlit as st
import pandas as pd
from shared import (
    page_hero, finding_box,
    REPORT_FINDINGS, build_player_report_pdf, build_team_report_pdf,
)


def show():
    page_hero(
        "Conclusions",
        "Final Takeaways & Decision Guide",
        "2025 Texas Rangers가 실제 81승에 그친 이유를 경기 운영, 선수 상태, 하이 레버리지 부진 케이스, 시뮬레이션 후보를 묶어 정리합니다.",
        [("Residual Summary", "white"), ("Decision Candidates", "white"), ("Team PDF", "white")],
    )

    finding_box(
        "이 페이지의 핵심",
        "결론은 '투수 폼 하나가 문제였다'가 아닙니다. 실제 81승과 기대 승수 90.06승 사이의 -9.06승 차이는 접전 경기, 불펜 운영, 선수별 상태, 의사결정 후보를 함께 봐야 설명됩니다. 모션 분석은 그중 선수별 원인을 더 자세히 보는 보조 근거입니다."
    )

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("""
        <div class="glass-card glass-card-navy">
            <div class="chart-title">폼 차이가 뚜렷한 케이스</div>
            <div class="chart-caption">
                <b>Webb</b>은 삼진 상황과 볼넷 상황에서 몸의 움직임 차이가 가장 뚜렷했습니다.<br><br>
                쉽게 말하면, 좋은 결과가 날 때는 골반과 어깨가 더 잘 분리되고 몸통 회전이 더 자연스럽게 이어졌습니다.
                따라서 Webb은 코칭이나 경기 전 루틴 조정으로 개선 여지를 검토할 수 있습니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="glass-card glass-card-red">
            <div class="chart-title">폼만으로 설명하기 어려운 케이스</div>
            <div class="chart-caption">
                <b>Garcia</b>와 <b>Leiter</b>는 결과가 좋을 때와 나쁠 때의 폼 차이가 크지 않았습니다.<br><br>
                이런 경우에는 폼 교정보다 구종 선택, 공의 위치, 상대 타자와의 매치업, 등판 상황을 먼저 확인하는 것이 더 적절합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">Jackson 해석</div>
        <div class="chart-caption">
            Jackson은 문제 있는 사이드암 투수라기보다, 몸을 옆으로 조금 기울여 던지는 오버핸드 패턴에 가깝습니다.
            따라서 큰 폼 교정보다는 좌타자/우타자 상대 성적을 확인하고, 강점이 있는 매치업에 맞춰 쓰는 방향이 더 현실적입니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="glass-card"><div class="chart-title">선수별 운영 권고</div>'
        '<div class="chart-caption">각 선수에게 같은 처방을 적용하지 않고, 코칭이 필요한 선수와 기용 방식 조정이 필요한 선수를 나눠 봅니다.</div></div>',
        unsafe_allow_html=True,
    )
    rec_df = pd.DataFrame({
        "선수": ["Leiter", "Webb", "Garcia", "Armstrong", "Jackson"],
        "핵심 해석": [
            "폼 차이는 제한적. 제구와 구종 선택 점검 필요",
            "폼 차이가 가장 뚜렷함. 코칭 개입 가능성 높음",
            "폼 차이는 작음. 마무리 기용 방식 점검 필요",
            "확정적 결론은 어려움. 불펜 루틴과 피로도 확인 필요",
            "폼 교정보다 매치업별 활용 방식 조정 필요",
        ],
        "권고": [
            "카운트별 구종 조합과 제구 위치 점검",
            "상체-하체 분리와 릴리스 타이밍 안정화",
            "상대 타순, 연투 여부, 좌우 매치업 기준 배치 조정",
            "등판 전 준비 시간과 연투 후 성적 관리",
            "좌우 타자별 강점 확인 후 제한적 활용",
        ],
        "우선순위": ["중간", "높음", "높음", "중간", "중간"],
    })
    st.dataframe(rec_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="glass-card glass-card-accent">
        <div class="chart-title">구단 의사결정과의 연결</div>
        <div class="chart-caption">
            이 분석은 "누구의 폼을 고칠 것인가"만 답하는 자료가 아닙니다.
            어떤 선수는 코칭이 필요하고, 어떤 선수는 기용 상황을 바꿔야 하며, 어떤 경우에는 외부 보강을 검토해야 합니다.<br><br>
            따라서 최종 판단은 선수별 해석과 Simulation 페이지의 의사결정 후보 순위표를 함께 보고 내려야 합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">시뮬레이션 결론</div>
        <div class="chart-caption">
            Simulation 페이지의 핵심은 여러 개선 후보를 같은 기준으로 비교하는 것입니다.
            수동 시나리오, Grid 후보, Pareto 후보를 비교해 승수 개선 폭이 큰 안과 예측이 안정적인 안을 나눠 볼 수 있습니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("분석 한계와 다음 단계", expanded=False):
        st.markdown("""
        <div style="font-size:15px; line-height:1.82; color:#475569;">

        <b>데이터 한계</b><br>
        - 선수별 영상 표본이 많지 않아, 모든 차이를 확정적으로 말하기는 어렵습니다.<br>
        - 방송 영상 기반이므로 카메라 각도와 프레임 수의 한계가 있습니다.<br><br>

        <b>모델 한계</b><br>
        - MotionAGFormer는 야구 전용 모델이 아니기 때문에, 수치는 의사결정의 보조 근거로 보는 것이 적절합니다.<br><br>

        <b>다음 단계</b><br>
        - 구종별로 나눠 다시 비교하기<br>
        - 시즌 전체 데이터를 길게 추적하기<br>
        - 좌타자/우타자 상대 성적과 등판 상황을 함께 연결하기

        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quote-card">
        <p><strong>종합 결론</strong></p>
        <p>
        TEX 2025의 -9.06승 잔차는 한 가지 이유로 설명하기 어렵습니다.
        Webb처럼 폼 조정 가능성이 보이는 선수도 있지만, Garcia나 Jackson처럼 기용 방식과 매치업을 먼저 봐야 하는 선수도 있습니다.<br>
        따라서 최종 권고는 <b>불펜 운영 개선, 선수별 코칭, 매치업 기반 기용, 필요 시 외부 보강</b>을 함께 검토하는 것입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">보고서 출력</div>
        <div class="chart-caption">
            선수별 보고서와 팀 전체 보고서를 PDF로 출력합니다. 보고서에는 잔차 해석, 선수별 근거, 시뮬레이션 후보가 같은 흐름으로 정리됩니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_col_1, report_col_2 = st.columns([1, 1], gap="large")
    with report_col_1:
        st.markdown("""
        <div class="glass-card glass-card-amber">
            <div class="chart-title">선수별 요약 보고서</div>
            <div class="chart-caption">
                선택한 선수의 핵심 지표, 쉬운 해석, 코칭 또는 기용 판단 포인트를 정리합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
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
        <div class="glass-card glass-card-amber">
            <div class="chart-title">팀 요약 보고서</div>
            <div class="chart-caption">
                팀 잔차 요약, 팀 장단점, 선수별 요약, 최종 결론을 한 문서로 정리합니다.
            </div>
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
