import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shared import (
    data, page_hero, kpi_card, section_header, fmt_num,
    TEX_BLUE, TEX_RED,
)


def show():
    page_hero(
        "Comparison",
        "Representative Pitcher Motion Layer",
        "잔차 분석의 하위 근거로 선정한 5명 투수의 상황별 키네마틱 차이를 Cohen's d와 p-value 기준으로 비교합니다. 필요한 지표는 필터로 직접 확인할 수 있게 구성했습니다.",
        [("Cohen's d", "white"), ("p-value", "white"), ("Interactive", "white")],
    )

    df = data['pitcher_ag'].copy()
    available_metrics = df['label'].dropna().unique().tolist()
    available_players = df['player'].dropna().unique().tolist()

    def _sig_label(p_value: float) -> str:
        if pd.isna(p_value):
            return "n/a"
        if p_value < 0.001:
            return "p<0.001"
        if p_value < 0.01:
            return "p<0.01"
        if p_value < 0.05:
            return "p<0.05"
        if p_value < 0.10:
            return "p<0.10"
        return "n.s."

    def _sig_group(p_value: float) -> str:
        if pd.isna(p_value):
            return "No p-value"
        if p_value < 0.01:
            return "Strong"
        if p_value < 0.05:
            return "Significant"
        if p_value < 0.10:
            return "Marginal"
        return "Not significant"

    chart_col_1, chart_col_2 = st.columns([1.08, 1], gap="large")

    with chart_col_1:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="chart-title">지표별 효과 크기</div>'
                '<div class="chart-caption">선택한 지표에서 투수별 Cohen&#39;s d를 비교합니다. 점선은 |d|=0.8 기준입니다.</div></div>',
                unsafe_allow_html=True
            )
            selected_metric = st.selectbox(
                "지표 선택",
                available_metrics,
                index=available_metrics.index('HSS @ FP (°)') if 'HSS @ FP (°)' in available_metrics else 0,
                key="comparison_metric_select_compact",
            )
            selected_players_bar = st.multiselect(
                "표시할 투수",
                available_players,
                default=available_players,
                key="comparison_bar_players_compact",
            )
            metric_df = df[df['label'] == selected_metric].copy()
            if selected_players_bar:
                metric_df = metric_df[metric_df['player'].isin(selected_players_bar)].copy()
            metric_df['cohens_d'] = pd.to_numeric(metric_df['cohens_d'], errors='coerce')
            metric_df['u_p'] = pd.to_numeric(metric_df['u_p'], errors='coerce')
            metric_df['abs_d'] = metric_df['cohens_d'].abs()
            metric_df['sig_label'] = metric_df['u_p'].map(_sig_label)
            metric_df['sig_group'] = metric_df['u_p'].map(_sig_group)
            metric_df = metric_df.sort_values('abs_d', ascending=True)

            if metric_df.empty:
                st.info("선택한 조건에 해당하는 지표 데이터가 없습니다.")
            else:
                color_map = {
                    "Strong": TEX_RED,
                    "Significant": "#D04A52",
                    "Marginal": "#F59F00",
                    "Not significant": "#9AA4B2",
                    "No p-value": "#CBD5E1",
                }
                metric_df['bar_text'] = metric_df['cohens_d'].map(lambda v: f"{v:+.2f}")
                fig_effect = px.bar(
                    metric_df,
                    x='cohens_d',
                    y='player',
                    orientation='h',
                    color='sig_group',
                    color_discrete_map=color_map,
                    text='bar_text',
                    custom_data=['abs_d', 'u_p', 'a_mean', 'b_mean', 'diff', 'sig_label'],
                    labels={'cohens_d': "Cohen's d", 'player': "Pitcher", 'sig_group': "Significance"},
                )
                fig_effect.update_traces(
                    textposition='inside',
                    insidetextanchor='middle',
                    cliponaxis=False,
                    marker_line_color='rgba(255,255,255,0.9)',
                    marker_line_width=1.0,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        + f"Metric: {selected_metric}<br>"
                        + "Cohen's d: %{x:+.3f}<br>"
                        + "|d|: %{customdata[0]:.3f}<br>"
                        + "p-value: %{customdata[1]:.4f}<br>"
                        + "A mean: %{customdata[2]:.3f}<br>"
                        + "B mean: %{customdata[3]:.3f}<br>"
                        + "Diff: %{customdata[4]:+.3f}<br>"
                        + "Significance: %{customdata[5]}<extra></extra>"
                    ),
                )
                x_min = float(metric_df['cohens_d'].min())
                x_max = float(metric_df['cohens_d'].max())
                pad = max(0.35, (x_max - x_min) * 0.16)
                fig_effect.add_vline(x=0, line_width=1.2, line_color='rgba(13,27,51,0.48)')
                fig_effect.add_vline(x=0.8, line_width=1, line_dash='dash', line_color='rgba(13,27,51,0.25)')
                fig_effect.add_vline(x=-0.8, line_width=1, line_dash='dash', line_color='rgba(13,27,51,0.25)')
                fig_effect.update_layout(
                    height=318,
                    margin=dict(l=8, r=16, t=14, b=34),
                    xaxis=dict(title="Cohen's d", range=[x_min - pad, x_max + pad], zeroline=False, gridcolor='rgba(13,27,51,0.08)'),
                    yaxis=dict(title=None, categoryorder='array', categoryarray=metric_df['player'].tolist()),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.28, xanchor='left', x=0, title=None, font=dict(size=10)),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family='Manrope, Arial, sans-serif', color='#1B2435', size=11),
                    bargap=0.36,
                )
                st.plotly_chart(fig_effect, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with chart_col_2:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="chart-title">키네마틱 프로필 레이더</div>'
                '<div class="chart-caption">선수별 라인만 표시합니다. 면 채움 없이 축별 분기 폭을 비교합니다.</div></div>',
                unsafe_allow_html=True
            )
            selected_players_radar = st.multiselect(
                "레이더 표시 투수",
                available_players,
                default=available_players,
                key="comparison_radar_players_compact",
            )
            selected_metrics_radar = st.multiselect(
                "레이더 지표",
                available_metrics,
                default=available_metrics,
                key="comparison_radar_metrics_compact",
            )
            radar_value_mode = st.radio(
                "값 기준",
                ["절대값 |d|", "부호 포함 d"],
                horizontal=True,
                key="comparison_radar_mode_compact",
            )

            radar_df = df[df['player'].isin(selected_players_radar) & df['label'].isin(selected_metrics_radar)].copy()
            radar_df['cohens_d'] = pd.to_numeric(radar_df['cohens_d'], errors='coerce')
            radar_df['u_p'] = pd.to_numeric(radar_df['u_p'], errors='coerce')
            radar_df['radar_value'] = radar_df['cohens_d'].abs() if radar_value_mode == "절대값 |d|" else radar_df['cohens_d']
            radar_df['sig_label'] = radar_df['u_p'].map(_sig_label)

            if radar_df.empty or not selected_metrics_radar or not selected_players_radar:
                st.info("레이더 차트를 그리려면 투수와 지표를 1개 이상 선택해 주세요.")
            else:
                metric_order = selected_metrics_radar
                radar_palette = [TEX_BLUE, TEX_RED, '#2F9E65', '#F59F00', '#7C3AED', '#0EA5E9', '#64748B']
                fig_radar = go.Figure()
                for idx, player in enumerate(selected_players_radar):
                    player_frame = radar_df[radar_df['player'] == player].set_index('label')
                    values, hover_lines = [], []
                    for metric in metric_order:
                        if metric in player_frame.index:
                            row = player_frame.loc[metric]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            value = float(row['radar_value']) if pd.notna(row['radar_value']) else 0.0
                            original_d = float(row['cohens_d']) if pd.notna(row['cohens_d']) else 0.0
                            p_value = float(row['u_p']) if pd.notna(row['u_p']) else np.nan
                            hover_lines.append(f"{metric}<br>d: {original_d:+.3f}<br>|d|: {abs(original_d):.3f}<br>p: {p_value:.4f}")
                        else:
                            value = 0.0
                            hover_lines.append(f"{metric}<br>No data")
                        values.append(value)
                    closed_metrics = metric_order + [metric_order[0]]
                    closed_values = values + [values[0]]
                    closed_hover = hover_lines + [hover_lines[0]]
                    color = radar_palette[idx % len(radar_palette)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=closed_values,
                        theta=closed_metrics,
                        mode='lines+markers',
                        name=player,
                        line=dict(color=color, width=2.6),
                        marker=dict(size=5, color=color),
                        fill=None,
                        opacity=1.0,
                        customdata=closed_hover,
                        hovertemplate="<b>%{fullData.name}</b><br>%{customdata}<extra></extra>",
                    ))
                max_abs = float(radar_df['radar_value'].abs().max()) if not radar_df.empty else 1.0
                radial_max = max(1.0, max_abs * 1.12)
                radial_min = -radial_max if radar_value_mode == "부호 포함 d" else 0
                fig_radar.update_layout(
                    height=360,
                    margin=dict(l=8, r=8, t=8, b=8),
                    polar=dict(
                        bgcolor='rgba(255,255,255,0)',
                        radialaxis=dict(visible=True, range=[radial_min, radial_max], gridcolor='rgba(13,27,51,0.10)', linecolor='rgba(13,27,51,0.18)', tickfont=dict(size=9, color='#64748B')),
                        angularaxis=dict(gridcolor='rgba(13,27,51,0.08)', linecolor='rgba(13,27,51,0.14)', tickfont=dict(size=9, color='#1B2435')),
                    ),
                    legend=dict(orientation='h', yanchor='bottom', y=-0.15, xanchor='left', x=0, title=None, font=dict(size=10)),
                    plot_bgcolor='rgba(255,255,255,0)',
                    paper_bgcolor='rgba(255,255,255,0)',
                    font=dict(family='Manrope, Arial, sans-serif', color='#1B2435', size=11),
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    if not df.empty:
        summary_df = (
            df.assign(abs_d=pd.to_numeric(df['cohens_d'], errors='coerce').abs(), u_p=pd.to_numeric(df['u_p'], errors='coerce'))
            .groupby('player', as_index=False)
            .agg(avg_abs_d=('abs_d', 'mean'), max_abs_d=('abs_d', 'max'), sig_count=('u_p', lambda s: int((s < 0.05).sum())))
            .sort_values(['avg_abs_d', 'max_abs_d'], ascending=False)
        )
        s1, s2, s3, s4, s5 = st.columns(5)
        for col, (_, row) in zip([s1, s2, s3, s4, s5], summary_df.iterrows()):
            with col:
                kpi_card(str(row['player']), fmt_num(row['avg_abs_d'], 2), f"max |d| {fmt_num(row['max_abs_d'], 2)} · sig {int(row['sig_count'])}", accent="red" if int(row['sig_count']) > 0 else "navy")

    section_header("투수별 종합 요약", "모든 지표를 표로 비교합니다. 진한 색일수록 효과 크기 또는 유의성이 큽니다.")
    pivot_d = df.pivot_table(index='player', columns='label', values='cohens_d').round(2)
    pivot_p = df.pivot_table(index='player', columns='label', values='u_p').round(3)

    def color_d(val):
        if pd.isna(val):
            return ''
        abs_d = abs(val)
        if abs_d > 1.5:
            return 'background-color: #003278; color: white; font-weight: bold'
        if abs_d > 0.8:
            return 'background-color: #6080B0; color: white'
        if abs_d > 0.5:
            return 'background-color: #B8C8E0'
        return ''

    def color_p(val):
        if pd.isna(val):
            return ''
        if val < 0.01:
            return 'background-color: #C0111F; color: white; font-weight: bold'
        if val < 0.05:
            return 'background-color: #E08080'
        if val < 0.10:
            return 'background-color: #FFE4B5'
        return ''

    table_tab_1, table_tab_2 = st.tabs(["Cohen's d", "p-value"])
    with table_tab_1:
        st.dataframe(pivot_d.style.map(color_d).format("{:+.2f}", na_rep="—"), use_container_width=True)
        st.caption("진한 파랑: |d|>1.5 | 파랑: |d|>0.8 | 연파랑: |d|>0.5")
    with table_tab_2:
        st.dataframe(pivot_p.style.map(color_p).format("{:.3f}", na_rep="—"), use_container_width=True)
        st.caption("빨강: p<0.01 | 연빨강: p<0.05 | 연주황: p<0.10")

    section_header("투수별 핵심 발견", "폼 교정이 필요한 선수와 운영·매치업 관점으로 봐야 하는 선수를 구분합니다.")
    summary_cards = [
        ("Leiter", "선발", "ns 대부분", "폼 자체는 일관, 결과 차이는 다른 요인", "🟡"),
        ("Webb", "선발", "강한 차이 ⭐", "HSS, Trunk/Hip ratio에서 매우 명확", "🔴"),
        ("Garcia", "마무리", "Null finding", "폼 일관됨, 블론 세이브는 외부 요인 가능성", "🟡"),
        ("Armstrong", "불펜", "지표 변동", "p<0.05 일부 지표 존재", "🟠"),
        ("Jackson", "불펜", "재해석", "Sidearm 아닌 Lateral tilt overhand", "🟢"),
    ]
    html = '<div class="inline-stat-grid">'
    for name, role, finding, note, marker in summary_cards:
        html += (
            f'<div class="pitcher-mini-card">'
            f'<div class="pitcher-mini-name">{marker} {name}</div>'
            f'<div class="pitcher-mini-role">{role}</div>'
            f'<div class="pitcher-mini-tag">{finding}</div>'
            f'<div class="pitcher-mini-note">{note}</div>'
            f'</div>'
        )
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-shell glass-card-accent"><div class="section-heading">폼 교정 가능 영역</div><div class="section-copy">Webb: HSS 안정화 코칭<br>Armstrong: 등판 루틴 표준화<br><br><b>→ 코칭 스태프 개입 가치</b></div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="section-shell"><div class="section-heading">폼 외 요인 의심</div><div class="section-copy">Leiter: 피칭 디자인 검토<br>Garcia: Deployment / 매치업<br>Jackson: Specialist 활용<br><br><b>→ 운영 차원 결정</b></div></div>', unsafe_allow_html=True)