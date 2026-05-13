import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from shared import (
    data, page_hero, kpi_card, fmt_num,
    TEX_BLUE, TEX_RED,
    glossary_box, KINEMATIC_TERMS, BASEBALL_TERMS,
)


_HR = "<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>"


def show():
    page_hero(
        "Comparison",
        "주요 투수 모션 분석",
        "잔차 원인을 선수 단위로 좁히기 위해 선정한 5명 투수의 상황별 키네마틱 차이를 <b>효과 크기</b>와 <b>유의 수준</b> 기준으로 비교합니다.<br>이 페이지는 Simulation 의사결정 후보를 해석할 때 코칭 가능 영역과 운영·보강 영역을 구분하는 근거로 사용합니다.",
        [("효과 크기", "white"), ("유의 수준", "white"), ("Interactive", "white")],
    )

    df = data["pitcher_ag"].copy()
    available_metrics = df["label"].dropna().unique().tolist()
    available_players = df["player"].dropna().unique().tolist()

    glossary_box(
        "그래프 읽는 법",
        {
            "효과 크기 (Cohen's d)": BASEBALL_TERMS["Cohen's d"],
            "유의 수준 (p-value)": BASEBALL_TERMS["p-value"],
            "HSS @ FP": KINEMATIC_TERMS["HSS @ FP"],
            "HSS max": KINEMATIC_TERMS["HSS max"],
            "Trunk/Hip ratio": KINEMATIC_TERMS["Trunk/Hip ratio"],
        },
    )

    st.markdown(
        """
        <div style="background:#F8FAFC;border-left:4px solid #003278;padding:14px 18px;margin:8px 0 24px;border-radius:6px;font-size:13.5px;color:#1B2435;line-height:1.7;">
            <b style="color:#003278;">해석 가이드 — 효과 크기·유의 수준이 크다고 '좋다'는 뜻이 아닙니다</b><br>
            <b>효과 크기</b>(Cohen's d)와 <b>유의 수준</b>(p-value)은 두 상황(예: SO vs Walk, SV vs BS) 간 동작이 <b>얼마나 명확히 다른가</b>를 보여주는 <i>식별력</i> 지표입니다.<br><br>
            <b>효과 크기 큼 + 유의 수준 낮음(p가 작음)</b> &nbsp;→&nbsp; 동작 차이 명확 &nbsp;→&nbsp; 결과 차이를 <b>키네마틱 메커니즘</b>으로 설명 가능 &nbsp;→&nbsp; <b>코칭·훈련</b>으로 접근할 영역<br>
            <b>효과 크기 작음 / 유의 수준 높음(p가 큼)</b> &nbsp;→&nbsp; 동작은 비슷 &nbsp;→&nbsp; 결과 차이는 <b>외부 요인</b>(운·매치업·leverage)이 우세 &nbsp;→&nbsp; <b>운영·보강</b>(불펜 매치업·라인업)으로 접근할 영역
        </div>
        """,
        unsafe_allow_html=True,
    )

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
            return "데이터 부족"
        if p_value < 0.01:
            return "매우 명확 (p<0.01)"
        if p_value < 0.05:
            return "유의 (p<0.05)"
        if p_value < 0.10:
            return "경계 (p<0.10)"
        return "차이 미미 (n.s.)"

    def _hex_to_rgba(hex_color: str, alpha: float) -> str:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 1. 선수별 동작 차이 분석")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:20px;">
            <div class="section-copy">투수별 <b>효과 크기</b>와 <b>유의 수준</b>을 함께 보면서 어떤 동작 지표에서 차이가 크게 벌어지는지 확인합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chart_col_1, chart_col_2 = st.columns([1.08, 1], gap="large")

    with chart_col_1:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="chart-title">지표별 효과 크기</div>'
                '<div class="chart-caption">선택한 지표에서 투수별 <b>효과 크기</b>(Cohen\'s d)를 비교합니다. 점선은 |d|=0.8(큰 차이 기준선)입니다.</div></div>',
                unsafe_allow_html=True,
            )
            selected_metric = st.selectbox(
                "지표 선택",
                available_metrics,
                index=available_metrics.index("HSS @ FP (°)") if "HSS @ FP (°)" in available_metrics else 0,
                key="comparison_metric_select_compact",
            )
            selected_players_bar = st.multiselect(
                "표시할 투수",
                available_players,
                default=available_players,
                key="comparison_bar_players_compact",
            )
            metric_df = df[df["label"] == selected_metric].copy()
            if selected_players_bar:
                metric_df = metric_df[metric_df["player"].isin(selected_players_bar)].copy()
            metric_df["cohens_d"] = pd.to_numeric(metric_df["cohens_d"], errors="coerce")
            metric_df["u_p"] = pd.to_numeric(metric_df["u_p"], errors="coerce")
            metric_df["abs_d"] = metric_df["cohens_d"].abs()
            metric_df["sig_label"] = metric_df["u_p"].map(_sig_label)
            metric_df["sig_group"] = metric_df["u_p"].map(_sig_group)
            metric_df = metric_df.sort_values("abs_d", ascending=True)

            if metric_df.empty:
                st.info("선택한 조건에 해당하는 지표 데이터가 없습니다.")
            else:
                color_map = {
                    "매우 명확 (p<0.01)": TEX_RED,
                    "유의 (p<0.05)": "#D04A52",
                    "경계 (p<0.10)": "#F59F00",
                    "차이 미미 (n.s.)": "#9AA4B2",
                    "데이터 부족": "#CBD5E1",
                }
                metric_df["bar_text"] = metric_df["cohens_d"].map(lambda v: f"{v:+.2f}")
                fig_effect = px.bar(
                    metric_df,
                    x="cohens_d",
                    y="player",
                    orientation="h",
                    color="sig_group",
                    color_discrete_map=color_map,
                    text="bar_text",
                    custom_data=["abs_d", "u_p", "a_mean", "b_mean", "diff", "sig_label"],
                    labels={"cohens_d": "효과 크기 (Cohen's d)", "player": "투수", "sig_group": "통계 명확성"},
                )
                fig_effect.update_traces(
                    textposition="inside",
                    insidetextanchor="middle",
                    cliponaxis=False,
                    marker_line_color="rgba(255,255,255,0.9)",
                    marker_line_width=1.0,
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        + f"지표: {selected_metric}<br>"
                        + "효과 크기 (d): %{x:+.3f}<br>"
                        + "|d|: %{customdata[0]:.3f}<br>"
                        + "유의 수준 (p): %{customdata[1]:.4f}<br>"
                        + "A 평균: %{customdata[2]:.3f}<br>"
                        + "B 평균: %{customdata[3]:.3f}<br>"
                        + "차이: %{customdata[4]:+.3f}<br>"
                        + "통계 명확성: %{customdata[5]}<extra></extra>"
                    ),
                )
                x_min = float(metric_df["cohens_d"].min())
                x_max = float(metric_df["cohens_d"].max())
                pad = max(0.35, (x_max - x_min) * 0.16)
                fig_effect.add_vline(x=0, line_width=1.2, line_color="rgba(13,27,51,0.48)")
                fig_effect.add_vline(x=0.8, line_width=1, line_dash="dash", line_color="rgba(13,27,51,0.25)")
                fig_effect.add_vline(x=-0.8, line_width=1, line_dash="dash", line_color="rgba(13,27,51,0.25)")
                fig_effect.update_layout(
                    height=318,
                    margin=dict(l=8, r=16, t=14, b=34),
                    xaxis=dict(
                        title="효과 크기 (Cohen's d)",
                        range=[x_min - pad, x_max + pad],
                        zeroline=False,
                        gridcolor="rgba(13,27,51,0.08)",
                    ),
                    yaxis=dict(title=None, categoryorder="array", categoryarray=metric_df["player"].tolist()),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.28,
                        xanchor="left",
                        x=0,
                        title=None,
                        font=dict(size=10),
                    ),
                    plot_bgcolor="rgba(255,255,255,0)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="Manrope, Arial, sans-serif", color="#1B2435", size=11),
                    bargap=0.36,
                )
                st.plotly_chart(fig_effect, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    with chart_col_2:
        with st.container():
            st.markdown(
                '<div class="glass-card"><div class="chart-title">키네마틱 프로필 레이더</div>'
                '<div class="chart-caption">선수별 라인만 표시합니다. 면 채움 없이 축별 분기 폭을 비교합니다.</div></div>',
                unsafe_allow_html=True,
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

            radar_df = df[df["player"].isin(selected_players_radar) & df["label"].isin(selected_metrics_radar)].copy()
            radar_df["cohens_d"] = pd.to_numeric(radar_df["cohens_d"], errors="coerce")
            radar_df["u_p"] = pd.to_numeric(radar_df["u_p"], errors="coerce")
            radar_df["radar_value"] = radar_df["cohens_d"].abs() if radar_value_mode == "절대값 |d|" else radar_df["cohens_d"]
            radar_df["sig_label"] = radar_df["u_p"].map(_sig_label)

            if radar_df.empty or not selected_metrics_radar or not selected_players_radar:
                st.info("레이더 차트를 그리려면 투수와 지표를 1개 이상 선택해 주세요.")
            else:
                metric_order = selected_metrics_radar
                radar_palette = [TEX_BLUE, TEX_RED, "#2F9E65", "#F59F00", "#7C3AED", "#0EA5E9", "#64748B"]
                fig_radar = go.Figure()
                for idx, player in enumerate(selected_players_radar):
                    player_frame = radar_df[radar_df["player"] == player].set_index("label")
                    values, hover_lines = [], []
                    for metric in metric_order:
                        if metric in player_frame.index:
                            row = player_frame.loc[metric]
                            if isinstance(row, pd.DataFrame):
                                row = row.iloc[0]
                            value = float(row["radar_value"]) if pd.notna(row["radar_value"]) else 0.0
                            original_d = float(row["cohens_d"]) if pd.notna(row["cohens_d"]) else 0.0
                            p_value = float(row["u_p"]) if pd.notna(row["u_p"]) else np.nan
                            hover_lines.append(
                                f"{metric}<br>d: {original_d:+.3f}<br>|d|: {abs(original_d):.3f}<br>p: {p_value:.4f}"
                            )
                        else:
                            value = 0.0
                            hover_lines.append(f"{metric}<br>No data")
                        values.append(value)
                    closed_metrics = metric_order + [metric_order[0]]
                    closed_values = values + [values[0]]
                    closed_hover = hover_lines + [hover_lines[0]]
                    color = radar_palette[idx % len(radar_palette)]
                    fig_radar.add_trace(
                        go.Scatterpolar(
                            r=closed_values,
                            theta=closed_metrics,
                            mode="lines+markers",
                            name=player,
                            line=dict(color=color, width=2.6),
                            marker=dict(size=5, color=color),
                            fill="toself",
                            fillcolor=_hex_to_rgba(color, 0.10),
                            opacity=1.0,
                            customdata=closed_hover,
                            hovertemplate="<b>%{fullData.name}</b><br>%{customdata}<extra></extra>",
                        )
                    )
                max_abs = float(radar_df["radar_value"].abs().max()) if not radar_df.empty else 1.0
                radial_max = max(1.0, max_abs * 1.12)
                radial_min = -radial_max if radar_value_mode == "부호 포함 d" else 0
                fig_radar.update_layout(
                    height=360,
                    margin=dict(l=8, r=8, t=8, b=8),
                    polar=dict(
                        bgcolor="rgba(255,255,255,0)",
                        radialaxis=dict(
                            visible=True,
                            range=[radial_min, radial_max],
                            gridcolor="rgba(13,27,51,0.10)",
                            linecolor="rgba(13,27,51,0.18)",
                            tickfont=dict(size=9, color="#64748B"),
                        ),
                        angularaxis=dict(
                            gridcolor="rgba(13,27,51,0.08)",
                            linecolor="rgba(13,27,51,0.14)",
                            tickfont=dict(size=9, color="#1B2435"),
                        ),
                    ),
                    legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="left", x=0, title=None, font=dict(size=10)),
                    plot_bgcolor="rgba(255,255,255,0)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    font=dict(family="Manrope, Arial, sans-serif", color="#1B2435", size=11),
                )
                st.plotly_chart(fig_radar, use_container_width=True, config={"displayModeBar": False, "responsive": True})

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 2. 선수별 요약 지표")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:18px;">
            <div class="section-copy">평균 |d|는 전체 동작 지표에서 차이가 얼마나 크게 나타났는지, sig는 p&lt;0.05인 지표 수를 의미합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_df = (
        df.assign(
            abs_d=pd.to_numeric(df["cohens_d"], errors="coerce").abs(),
            u_p=pd.to_numeric(df["u_p"], errors="coerce"),
        )
        .groupby("player", as_index=False)
        .agg(
            avg_abs_d=("abs_d", "mean"),
            max_abs_d=("abs_d", "max"),
            sig_count=("u_p", lambda s: int((s < 0.05).sum())),
        )
        .sort_values(["avg_abs_d", "max_abs_d"], ascending=False)
    )
    summary_cols = st.columns(5)
    for col, (_, row) in zip(summary_cols, summary_df.iterrows()):
        with col:
            kpi_card(
                str(row["player"]),
                fmt_num(row["avg_abs_d"], 2),
                f"max |d| {fmt_num(row['max_abs_d'], 2)} · sig {int(row['sig_count'])}",
                accent="red" if int(row["sig_count"]) > 0 else "navy",
            )

    st.markdown(
        """
        <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;align-items:center;">
            <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                <span style="width:12px;height:12px;border-radius:3px;background:rgba(179,25,34,0.14);border:1px solid rgba(179,25,34,0.28);display:inline-block;"></span>
                p&lt;0.05 지표 1개 이상
            </span>
            <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                <span style="width:12px;height:12px;border-radius:3px;background:rgba(36,58,94,0.12);border:1px solid rgba(36,58,94,0.24);display:inline-block;"></span>
                p&lt;0.05 지표 없음
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 3. 투수별 종합 요약 표")
    st.markdown(
        """
        <div class="glass-card">
            <div class="section-copy">
                모든 동작 지표를 한 표에 비교합니다.<br>
                <b>진한 색</b> = 두 상황(SO/Walk, SV/BS) 간 동작이 <b>명확히 다름</b> → 결과 차이가 <b>동작 메커니즘으로 식별 가능</b> → <b>코칭·훈련으로 접근할 영역</b>.<br>
                <b>연한 색 / 빈 칸</b> = 동작은 비슷함 → 결과 차이는 <b>운·매치업·leverage 등 외부 요인</b>의 영향이 더 큼 → <b>운영·보강(불펜 매치업·라인업)으로 접근할 영역</b>.<br>
                <span style="color:#94A3B8;">※ 진한 색이 "좋다"는 뜻이 아니라, <b>어디서 결과 차이가 만들어지는가</b>를 보여주는 해석 기준입니다.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pivot_d = df.pivot_table(index="player", columns="label", values="cohens_d").round(2)
    pivot_p = df.pivot_table(index="player", columns="label", values="u_p").round(3)

    def color_d(val):
        if pd.isna(val):
            return ""
        abs_d = abs(val)
        if abs_d > 1.5:
            return "background-color: #003278; color: white; font-weight: bold"
        if abs_d > 0.8:
            return "background-color: #6080B0; color: white"
        if abs_d > 0.5:
            return "background-color: #B8C8E0"
        return ""

    def color_p(val):
        if pd.isna(val):
            return ""
        if val < 0.01:
            return "background-color: #C0111F; color: white; font-weight: bold"
        if val < 0.05:
            return "background-color: #E08080"
        if val < 0.10:
            return "background-color: #FFE4B5"
        return ""

    table_tab_1, table_tab_2 = st.tabs(["효과 크기 (Cohen's d)", "유의 수준 (p-value)"])
    with table_tab_1:
        st.dataframe(pivot_d.style.map(color_d).format("{:+.2f}", na_rep="—"), use_container_width=True)
        st.markdown(
            """
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;align-items:center;">
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#003278;display:inline-block;"></span>
                    |d|&gt;1.5
                </span>
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#6080B0;display:inline-block;"></span>
                    |d|&gt;0.8
                </span>
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#B8C8E0;display:inline-block;"></span>
                    |d|&gt;0.5
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with table_tab_2:
        st.dataframe(pivot_p.style.map(color_p).format("{:.3f}", na_rep="—"), use_container_width=True)
        st.markdown(
            """
            <div style="display:flex;gap:16px;flex-wrap:wrap;margin-top:8px;align-items:center;">
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#C0111F;display:inline-block;"></span>
                    p&lt;0.01
                </span>
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#E08080;display:inline-block;"></span>
                    p&lt;0.05
                </span>
                <span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#667085;">
                    <span style="width:12px;height:12px;border-radius:3px;background:#FFE4B5;display:inline-block;"></span>
                    p&lt;0.10
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 4. 투수별 핵심 발견")
    st.markdown(
        """
        <div class="glass-card" style="margin-bottom:18px;">
            <div class="section-copy">투수별 요약 지표와 표를 합쳐서 코칭 개입과 운용 판단을 나눠 봅니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _PITCHER_CARDS = [
        {
            "name": "Leiter",
            "type": "선발",
            "type_color": "#003278",
            "bg": "rgba(240,245,255,0.97)",
            "border": "rgba(0,50,120,0.22)",
            "finding": "투구 폼 변화 없음",
            "action": "상황과 무관하게 자세가 일정함<br>성적 차이는 폼 외 다른 원인",
        },
        {
            "name": "Webb",
            "type": "선발",
            "type_color": "#B31922",
            "bg": "rgba(255,244,244,0.97)",
            "border": "rgba(179,25,34,0.22)",
            "finding": "동작 차이 뚜렷",
            "action": "어깨 회전 속도·상하체 협응에서<br>상황별 폼 변화가 통계적으로 명확",
        },
        {
            "name": "Garcia",
            "type": "마무리",
            "type_color": "#9A3412",
            "bg": "rgba(255,247,237,0.97)",
            "border": "rgba(154,52,18,0.22)",
            "finding": "투구 폼 변화 없음",
            "action": "자세는 안정적<br>세이브 실패는 상대·상황 등 외부 요인 가능성",
        },
        {
            "name": "Armstrong",
            "type": "불펜",
            "type_color": "#166534",
            "bg": "rgba(240,253,244,0.97)",
            "border": "rgba(22,101,52,0.22)",
            "finding": "일부 동작 불안정",
            "action": "등판 상황에 따라<br>특정 동작에서 유의미한 변화 발생",
        },
        {
            "name": "Jackson",
            "type": "불펜",
            "type_color": "#4A5568",
            "bg": "rgba(247,250,252,0.97)",
            "border": "rgba(74,85,104,0.22)",
            "finding": "투구 폼 재해석",
            "action": "옆으로 던지는 것처럼 보이지만<br>실제로는 상체를 기울인 정통 오버핸드",
        },
    ]

    case_cols = st.columns(5, gap="small")
    for col, card in zip(case_cols, _PITCHER_CARDS):
        with col:
            st.markdown(
                f"""
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
                    {card['action']}
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 5. 의사결정 방향")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown(
            '<div class="glass-card glass-card-navy"><div class="chart-title">투구 폼 교정으로 개선 가능</div>'
            '<div class="chart-caption">Webb: 어깨·골반 회전 타이밍 교정<br>Armstrong: 등판 전 준비 루틴 일관화<br><br><b>→ 코칭 스태프가 직접 개입할 수 있는 영역</b></div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="glass-card glass-card-red"><div class="chart-title">투구 폼 이외의 원인 검토 필요</div>'
            '<div class="chart-caption">Leiter: 구종·배합 전략 재검토<br>Garcia: 기용 시점 및 상대 타자 배치 재검토<br>Jackson: 특정 상황 전담 투수로 활용<br><br><b>→ 감독·코치진의 운영 전략 결정 영역</b></div></div>',
            unsafe_allow_html=True,
        )
