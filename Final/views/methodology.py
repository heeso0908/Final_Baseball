import streamlit as st
from shared import data, ASSETS, page_hero, finding_box, glossary_box, KINEMATIC_TERMS

_HR = "<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>"


def show():
    page_hero(
        "Methodology",
        "AI 동작 분석 모델 선택 근거",
        """승수 차이의 원인을 경기 운영과 선수 성적으로 좁힌 뒤, 결정적 순간에 부진했던 투수의 투구 동작을 검증하기 위해 두 가지 AI 모델(MotionBERT, MotionAGFormer)의 측정 안정성을 비교했습니다.<br>
        더 일관된 결과를 내는 MotionAGFormer를 최종 채택해 투구 동작 분석에 사용합니다.""",
        [("MotionBERT", "white"), ("MotionAGFormer", "white"), ("측정 안정성", "white")],
    )

    df_model = data['model_sum'].pivot_table(
        index='metric', columns='model', values='cv_pct'
    ).reset_index()

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 1. 키네마틱 지표 용어")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:18px;">
            <div class="section-copy">표와 그래프를 읽기 전에 필요한 용어만 짧게 정리했습니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    glossary_box("키네마틱 지표 용어", KINEMATIC_TERMS, mb=0, show_header=False)

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 2. 모델 및 지표 검증")
    top_left, top_right = st.columns([1.1, 1], gap="large")

    with top_left:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="section-heading">모델별 측정 안정성</div>'
                '<div class="section-copy">같은 동작을 반복 측정했을 때 결과가 얼마나 일관되는지 비교합니다. 흔들림이 작을수록 신뢰할 수 있는 모델입니다.<br>CV%(변동계수)가 낮을수록 안정적입니다.</div></div>',
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
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="section-heading">측정 지표 정의</div>'
                '<div class="section-copy">코칭으로 고칠 수 있는 문제인지 판단하기 위해, 투구 시 골반과 어깨의 분리·회전 속도·타이밍 차이를 3D로 측정합니다.</div></div>',
                unsafe_allow_html=True
            )
            import pandas as pd
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
            metrics_info = pd.DataFrame({
                '지표': ['HSS at FP', 'Hip peak 3D', 'Trunk peak 3D', 'Trunk/Hip ratio', 'Timing diff', 'HSS max'],
                '의미': [
                    '앞발이 땅에 닿는 순간의 골반-어깨 분리',
                    '골반 회전 속도의 최고값',
                    '몸통 회전 속도의 최고값',
                    '몸통 회전 속도와 골반 회전 속도의 비율',
                    '골반 회전 피크와 몸통 회전 피크 사이의 시간 차이',
                    '투구 동작 전체에서 가장 큰 골반-어깨 분리',
                ],
                '단위': ['도', '도/초', '도/초', '비율', 'ms', '도'],
            })
            st.dataframe(metrics_info, use_container_width=True, hide_index=True)
