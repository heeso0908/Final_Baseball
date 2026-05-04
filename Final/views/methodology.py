import streamlit as st
from shared import data, ASSETS, page_hero, finding_box, section_header


def show():
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
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="section-heading">모델별 측정 안정성</div>'
                '<div class="section-copy">Coefficient of Variation(CV%) 기준으로 낮을수록 반복 측정 안정성이 높습니다.</div></div>',
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
                '<div class="section-copy">투구 폼의 분리, 회전, 타이밍을 설명하는 핵심 3D 키네마틱 지표입니다.</div></div>',
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