import streamlit as st
from shared import data, ASSETS, page_hero, finding_box, glossary_box, KINEMATIC_TERMS


def show():
    page_hero(
        "Methodology",
        "Motion Analysis Pipeline",
        "잔차 원인을 경기 운영과 선수 상태로 좁힌 뒤, 하이 레버리지 상황에서 부진했던 투수 대표 케이스를 검증하기 위해 MotionBERT와 MotionAGFormer의 측정 안정성을 비교했습니다. 최종적으로 더 안정적인 MotionAGFormer 기반 키네마틱 지표를 모션 근거 레이어에 사용합니다.",
        [("MotionBERT", "white"), ("MotionAGFormer", "white"), ("CV Stability", "white")],
    )

    df_model = data['model_sum'].pivot_table(
        index='metric', columns='model', values='cv_pct'
    ).reset_index()

    glossary_box("키네마틱 지표 용어", KINEMATIC_TERMS)

    top_left, top_right = st.columns([1.1, 1], gap="large")

    with top_left:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="section-heading">모델별 측정 안정성</div>'
                '<div class="section-copy">반복 측정의 흔들림이 작은 모델을 선택해야 투수별 폼 차이를 의사결정 근거로 사용할 수 있습니다. CV%는 낮을수록 안정적입니다.</div></div>',
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
                '<div class="section-copy">코칭 가능 영역과 운영·매치업 이슈를 구분하기 위해 투구 폼의 분리, 회전, 타이밍을 설명하는 3D 키네마틱 지표를 사용합니다.</div></div>',
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
