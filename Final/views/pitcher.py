import streamlit as st
import pandas as pd
from shared import data, ASSETS, kpi_card, page_hero, glossary_box, KINEMATIC_TERMS, BASEBALL_TERMS


_SITUATION_LABEL = {
    "SO": "삼진", "BB": "볼넷", "Walk": "볼넷",
    "SV": "세이브", "BS": "블론 세이브",
}

def show_pitcher_page(pitcher_name, situation_a, situation_b,
                      interpretation_md, key_findings):
    label_a = _SITUATION_LABEL.get(situation_a, situation_a)
    label_b = _SITUATION_LABEL.get(situation_b, situation_b)
    role_kr = "선발" if pitcher_name in ["Leiter", "Webb"] else (
        "마무리" if pitcher_name == "Garcia" else "불펜"
    )
    page_hero(
        "Pitcher Detail",
        f"{pitcher_name} · {role_kr}",
        f"잔차 분석을 선수 단위로 좁히기 위해 {label_a}와 {label_b} 상황에서 3D pose 기반 키네마틱 지표와 영상 차이를 비교합니다.",
        [(f"{label_a} vs {label_b}", "white"), (role_kr, "white"), ("MotionAGFormer", "white")],
    )

    df = data['pitcher_ag']
    pitcher_df = df[df['player'] == pitcher_name]
    n_a = pitcher_df['n_a'].iloc[0] if len(pitcher_df) > 0 else 0
    n_b = pitcher_df['n_b'].iloc[0] if len(pitcher_df) > 0 else 0

    glossary_box("이 페이지의 주요 용어", {
        "HSS @ FP": KINEMATIC_TERMS["HSS @ FP"],
        "HSS max": KINEMATIC_TERMS["HSS max"],
        "Trunk/Hip ratio": KINEMATIC_TERMS["Trunk/Hip ratio"],
        "Cohen's d": BASEBALL_TERMS["Cohen's d"],
        "p-value": BASEBALL_TERMS["p-value"],
        "세이브(SV) / 블론 세이브(BS)": BASEBALL_TERMS["세이브(SV) / 블론 세이브(BS)"],
    })

    # ── 1. 영상 비교 ───────────────────────────────────────────
    st.markdown(
        '<div class="glass-card"><div class="section-heading">영상 비교</div>'
        '<div class="section-copy">원본과 스켈레톤 오버레이를 전환하며 같은 투수의 성공/실패 장면을 비교합니다.</div></div>',
        unsafe_allow_html=True
    )
    st.markdown("""
    <div class="video-mode-head">
        <strong>영상 모드 선택</strong>
        <span>원본 영상과 스켈레톤 오버레이를 전환합니다.</span>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio(
        "영상 모드",
        ["원본", "스켈레톤"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"{pitcher_name}_mode"
    )
    folder = "skeleton" if mode == "스켈레톤" else "original"
    case_a = f"{pitcher_name.lower()}_{situation_a.lower()}"
    case_b = f"{pitcher_name.lower()}_{situation_b.lower()}"
    video_a = ASSETS / "videos" / folder / f"{case_a}.mp4"
    video_b = ASSETS / "videos" / folder / f"{case_b}.mp4"

    with st.container(border=True):
        video_col_1, video_col_2 = st.columns(2, gap="medium")
        with video_col_1:
            st.markdown(
                f'<div class="video-case-title success">'
                f'<span class="case-icon"><i class="bi bi-check-square-fill"></i></span>'
                f'<span>{label_a}</span>'
                f'<span class="case-n">(n={n_a})</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            if video_a.exists():
                st.video(str(video_a), muted=True)
            else:
                st.warning(f"영상 없음: {case_a}.mp4")
        with video_col_2:
            st.markdown(
                f'<div class="video-case-title fail">'
                f'<span class="case-icon"><i class="bi bi-x-square-fill"></i></span>'
                f'<span>{label_b}</span>'
                f'<span class="case-n">(n={n_b})</span>'
                f'</div>',
                unsafe_allow_html=True
            )
            if video_b.exists():
                st.video(str(video_b), muted=True)
            else:
                st.warning(f"영상 없음: {case_b}.mp4")

    # ── 2. 핵심 키네마틱 지표 ──────────────────────────────────
    st.markdown(
        '<div class="glass-card"><div class="section-heading">핵심 키네마틱 지표</div>'
        '<div class="section-copy">Cohen&#39;s d 절대값이 큰 순서로 상위 지표를 요약했습니다.</div></div>',
        unsafe_allow_html=True
    )
    if key_findings:
        kpi_cols = st.columns(len(key_findings), gap="medium")
        for col, (label, mean_a, mean_b, p_val, d) in zip(kpi_cols, key_findings):
            with col:
                delta_str = f"{mean_a - mean_b:+.2f}"
                sig_text = "유의" if p_val < 0.05 else "n.s."
                accent = "red" if p_val < 0.05 else "navy"
                kpi_card(label, f"{mean_a:.2f}", f"Δ {delta_str} | p={p_val:.3f} | d={d:+.2f} ({sig_text})", accent=accent)

    # ── 3. 통계 검정 결과 ──────────────────────────────────────
    st.markdown(
        '<div class="glass-card"><div class="section-heading">통계 검정 결과</div>'
        '<div class="section-copy">p-value가 낮고 효과 크기 |d|가 클수록 상황별 폼 차이가 뚜렷합니다.</div></div>',
        unsafe_allow_html=True
    )
    if len(pitcher_df) > 0:
        display_df = pitcher_df[['label', 'a_mean', 'b_mean', 'diff', 'cohens_d', 'u_p', 't_p']].copy()
        display_df.columns = ['지표', f'{label_a} 평균', f'{label_b} 평균', '차이', "Cohen's d", 'p (Mann-Whitney)', 'p (t-test)']
        for col in display_df.columns[1:]:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce').round(3)

        def highlight_sig(row):
            try:
                p = float(row['p (Mann-Whitney)'])
                if p < 0.01:
                    return ['background-color: #FFE4E1'] * len(row)
                elif p < 0.05:
                    return ['background-color: #FFF8DC'] * len(row)
            except Exception:
                pass
            return [''] * len(row)

        st.dataframe(display_df.style.apply(highlight_sig, axis=1), use_container_width=True, hide_index=True)
        st.caption("🟥 p<0.01 | 🟨 p<0.05")
    else:
        st.info("해당 투수의 통계 데이터가 없습니다.")

    # ── 4. 분석 해석 ───────────────────────────────────────────
    st.markdown(
        f'<div class="glass-card"><div class="section-heading" style="margin-bottom:12px">분석 해석</div>\n\n{interpretation_md}\n\n</div>',
        unsafe_allow_html=True
    )
