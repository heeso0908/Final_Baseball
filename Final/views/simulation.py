import streamlit as st
import pandas as pd
import altair as alt
from shared import (
    data, RAW_DIR, SIMULATION_OPTIONS, DEFAULT_SCENARIO, DEFAULT_SIM_RUNS,
    kpi_card, section_badges, finding_box,
    get_simulation_result, get_scenario_snapshots,
    fmt_pct, fmt_num,
)


def show():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">Monte Carlo</span>
        <span class="pill pill-white">Residual Model</span>
        <span class="pill pill-white">2025 Season Rebuild</span>
        <h1>2025 TEX 시즌 시뮬레이션</h1>
        <p>
        2025 텍사스 레인저스의 실제 시즌을 기준으로 주요 전력 변수 변화가 승수에 미치는 영향을 재구성합니다.
        경기력 분석과 선수 분석에서 나온 가설을 가상 시나리오로 돌렸을 때 승수 분포와 residual layer가 어떻게 바뀌는지 확인합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_badges(
        ("Baseline / Bullpen / Rotation / Hitter Scenario", "red"),
        ("Actual 162-game Schedule", "navy"),
    )

    st.markdown("---")

    control_left, control_right = st.columns([1.2, 1], gap="large")

    with control_left:
        st.markdown("## 시나리오 실행")
        st.markdown("""
        <div class="glass-card">
            <b>실행 방식</b><br>
            시뮬레이션은 페이지 최초 진입 시 자동 실행되지 않습니다.<br>
            아래 조건을 선택한 뒤 버튼을 누르면 경기력·선수 조건 변화가 잔차와 승수에 미치는 결과를 계산합니다.
        </div>
        """, unsafe_allow_html=True)

    with control_right:
        st.markdown("## 조건 선택")
        selected_scenario = st.selectbox(
            "Scenario",
            SIMULATION_OPTIONS,
            index=SIMULATION_OPTIONS.index(st.session_state.get("sim_scenario", DEFAULT_SCENARIO)),
            key="simulation_scenario_select",
        )
        simulation_runs = st.slider(
            "Monte Carlo Runs",
            min_value=100,
            max_value=1000,
            value=int(st.session_state.get("sim_runs", DEFAULT_SIM_RUNS)),
            step=100,
            key="simulation_runs_slider",
        )
        run_click = st.button("Run Simulation", type="primary", use_container_width=True)

    if "simulation_result" not in st.session_state:
        st.session_state["simulation_result"] = None
        st.session_state["sim_scenario"] = DEFAULT_SCENARIO
        st.session_state["sim_runs"] = DEFAULT_SIM_RUNS

    if run_click:
        if not RAW_DIR.exists():
            st.error("data_raw 폴더가 없습니다. app.py/simulator.py에서 쓰던 CSV 파일들을 data_raw 폴더에 넣어주세요.")
            return
        try:
            with st.spinner("시뮬레이션 실행 중..."):
                st.session_state["simulation_result"] = get_simulation_result(
                    str(RAW_DIR), selected_scenario, simulation_runs,
                )
                st.session_state["sim_scenario"] = selected_scenario
                st.session_state["sim_runs"] = simulation_runs
        except FileNotFoundError as exc:
            st.error(f"필수 데이터 파일이 없습니다: {exc}")
            st.info("data_raw 폴더 안의 CSV 파일명과 simulator.py에서 요구하는 파일명이 같은지 확인해주세요.")
            return
        except Exception as exc:
            st.error(f"시뮬레이션 실행 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
            return

    result = st.session_state.get("simulation_result")

    if result is None:
        finding_box(
            "아직 시뮬레이션을 실행하지 않았습니다.",
            "상단에서 시나리오와 반복 횟수를 선택한 뒤 Run Simulation을 눌러주세요. 앱 첫 로딩 속도를 위해 계산은 lazy-load 방식으로 분리했습니다."
        )
        return

    summary = result.get("summary", {})
    distribution = result.get("distribution", pd.DataFrame()).copy()
    monthly = result.get("monthly_summary", pd.DataFrame()).copy()
    schedule_context = result.get("schedule_context", pd.DataFrame()).copy()
    players = result.get("player_projection", pd.DataFrame()).copy()
    pitchers = result.get("pitcher_projection", pd.DataFrame()).copy()

    st.markdown("---")
    st.markdown("## 시뮬레이션 결과 요약")

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Mean Wins", fmt_num(summary.get("mean")), f"Scenario: {st.session_state['sim_scenario']}", accent="red")
    with k2:
        kpi_card("Median Wins", fmt_num(summary.get("median")), f"Runs: {st.session_state['sim_runs']}", accent="navy")
    with k3:
        kpi_card("P10 - P90", f"{fmt_num(summary.get('p10'))} - {fmt_num(summary.get('p90'))}", "중앙 80% 결과 범위", accent="red")
    with k4:
        kpi_card("82+ Wins", fmt_pct(summary.get("over_81_5")), "0.500 이상 가능성", accent="navy")

    st.markdown("""
    <div class="finding-box">
        <strong>해석 기준</strong><br>
        이 결과는 미래 예측이라기보다 <b>2025 시즌을 조건 변화에 따라 다시 재구성한 결과</b>입니다.
        피타고리안 기대 승수와 실제 승수의 괴리를 설명하기 위해, 시나리오별 승수 분포와 residual 보정값을 함께 확인합니다.
    </div>
    """, unsafe_allow_html=True)

    chart_left, chart_right = st.columns([1.35, 1], gap="large")

    with chart_left:
        st.markdown("### 승수 분포")
        if not distribution.empty and "wins" in distribution.columns:
            win_chart = (
                alt.Chart(distribution)
                .mark_bar(color="#B31922", opacity=0.82, binSpacing=2)
                .encode(
                    x=alt.X("wins:Q", bin=alt.Bin(maxbins=18), title="Projected Wins"),
                    y=alt.Y("count():Q", title="Simulation Count"),
                    tooltip=[alt.Tooltip("count():Q", title="Count")],
                )
                .properties(height=330)
            )
            st.altair_chart(win_chart, use_container_width=True)
        else:
            st.info("승수 분포 데이터가 없습니다.")

    with chart_right:
        st.markdown("### Residual Layer")
        st.markdown(f"""
        <div class="glass-card glass-card-accent">
            <div class="kpi-label-custom">Active Scenario</div>
            <div class="kpi-value-custom" style="font-size:22px;">{st.session_state['sim_scenario']}</div>
            <div class="kpi-sub-custom">Monte Carlo {st.session_state['sim_runs']}회</div>
            <hr style="margin:14px 0;">
            <div style="font-size:13px; line-height:1.8; color:#344054;">
                <b>Residual bonus</b>: <span class="num">{fmt_num(summary.get('residual_bonus'), 2, sign=True)}</span> wins<br>
                <b>CV MAE</b>: <span class="num">{fmt_num(summary.get('ensemble_cv_mae'), 2)}</span><br>
                <b>Calibration offset</b>: <span class="num">{fmt_num(summary.get('calibration_offset'), 2, sign=True)}</span> wins<br>
                <b>88+ Wins</b>: <span class="num">{fmt_pct(summary.get('over_87_5'))}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 월별 시뮬레이션 흐름")
    monthly_left, monthly_right = st.columns([1.35, 1], gap="large")

    with monthly_left:
        if not monthly.empty and {"month", "mean_wins", "p25_wins", "p75_wins"}.issubset(monthly.columns):
            month_order = list(monthly["month"])
            monthly_band = (
                alt.Chart(monthly)
                .mark_area(opacity=0.18, color="#B31922")
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("p25_wins:Q", title="Projected Wins"),
                    y2="p75_wins:Q",
                    tooltip=[
                        alt.Tooltip("month:N", title="Month"),
                        alt.Tooltip("mean_wins:Q", title="Mean", format=".2f"),
                        alt.Tooltip("p25_wins:Q", title="P25", format=".2f"),
                        alt.Tooltip("p75_wins:Q", title="P75", format=".2f"),
                    ],
                )
            )
            monthly_line = (
                alt.Chart(monthly)
                .mark_line(point=True, strokeWidth=3, color="#0D1B33")
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("mean_wins:Q", title="Projected Wins"),
                )
            )
            st.altair_chart((monthly_band + monthly_line).properties(height=300), use_container_width=True)
        else:
            st.info("월별 요약 데이터가 없습니다.")

    with monthly_right:
        st.markdown("### Schedule Context")
        if not schedule_context.empty:
            rename_map = {
                "month": "Month",
                "games": "G",
                "home_games": "Home",
                "away_games": "Away",
                "strength_index": "Strength",
                "difficulty": "Difficulty",
            }
            display = schedule_context.rename(columns=rename_map)
            keep_cols = [col for col in ["Month", "G", "Home", "Away", "Strength", "Difficulty"] if col in display.columns]
            st.dataframe(display[keep_cols], use_container_width=True, hide_index=True)
        else:
            st.info("스케줄 요약 데이터가 없습니다.")

    st.markdown("---")
    hitter_tab, pitcher_tab, scenario_tab = st.tabs(["Hitters", "Pitchers", "Scenario Board"])

    with hitter_tab:
        st.markdown("### 타자 시나리오 카드")
        if not players.empty:
            card_players = players.head(4).copy()
            cols = st.columns(4)
            for col, (_, player) in zip(cols, card_players.iterrows()):
                with col:
                    kpi_card(
                        str(player.get("player", "-")),
                        fmt_num(player.get("sim_on_base"), 3),
                        f"{player.get('archetype', '-')} | HR {fmt_num(player.get('sim_hr'), 3)}",
                        accent="red" if bool(player.get("boosted", False)) else "navy",
                    )
            player_display = players.copy()
            for source, target, digits in [
                ("sim_on_base", "Modeled OBP", 3),
                ("sim_hr", "HR Rate", 3),
                ("sim_xbh", "XBH Rate", 3),
                ("sim_k", "K Rate", 3),
                ("delta_obp_pts", "OBP Delta", 1),
                ("delta_hr_pts", "HR Delta", 1),
            ]:
                if source in player_display.columns:
                    player_display[target] = player_display[source].map(lambda v: fmt_num(v, digits, sign=("Delta" in target)))
            keep = [c for c in ["player", "pos", "archetype", "Modeled OBP", "HR Rate", "XBH Rate", "K Rate", "OBP Delta", "HR Delta"] if c in player_display.columns]
            st.dataframe(player_display[keep].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("타자 projection 데이터가 없습니다.")

    with pitcher_tab:
        st.markdown("### 투수 시나리오 카드")
        if not pitchers.empty:
            card_pitchers = pitchers.head(4).copy()
            cols = st.columns(4)
            for col, (_, pitcher) in zip(cols, card_pitchers.iterrows()):
                delta_era = float(pitcher.get("delta_era", 0.0)) if pd.notna(pitcher.get("delta_era", 0.0)) else 0.0
                with col:
                    kpi_card(
                        str(pitcher.get("player", "-")),
                        fmt_num(pitcher.get("sim_era"), 2),
                        f"{pitcher.get('role', '-')} | ΔERA {fmt_num(delta_era, 2, sign=True)}",
                        accent="navy" if delta_era <= 0 else "red",
                    )
            pitcher_display = pitchers.copy()
            for source, target, digits in [
                ("sim_era", "Modeled ERA", 2),
                ("sim_whip", "Modeled WHIP", 2),
                ("sim_k9", "Modeled K/9", 1),
                ("delta_era", "ERA Delta", 2),
                ("sim_ip", "Projected IP", 1),
            ]:
                if source in pitcher_display.columns:
                    pitcher_display[target] = pitcher_display[source].map(lambda v: fmt_num(v, digits, sign=(target == "ERA Delta")))
            keep = [c for c in ["player", "role", "archetype", "Modeled ERA", "Modeled WHIP", "Modeled K/9", "ERA Delta", "Projected IP"] if c in pitcher_display.columns]
            st.dataframe(pitcher_display[keep].head(12), use_container_width=True, hide_index=True)
        else:
            st.info("투수 projection 데이터가 없습니다.")

    with scenario_tab:
        st.markdown("### 시나리오별 선수 변화 비교")
        try:
            snapshots = get_scenario_snapshots(str(RAW_DIR))
        except Exception as exc:
            st.warning(f"시나리오 보드 데이터를 불러오지 못했습니다: {type(exc).__name__}: {exc}")
            snapshots = None

        if snapshots:
            compare_mode = st.radio("Comparison Type", ["Hitters", "Pitchers"], horizontal=True)
            if compare_mode == "Hitters" and "hitters" in snapshots:
                hitter_snapshots = snapshots["hitters"].copy()
                hitter_names = hitter_snapshots["player"].drop_duplicates().tolist()
                default_idx = hitter_names.index("Wyatt Langford") if "Wyatt Langford" in hitter_names else 0
                selected_hitter = st.selectbox("Player", hitter_names, index=default_idx)
                hitter_compare = hitter_snapshots[hitter_snapshots["player"] == selected_hitter].copy()
                compare_chart = (
                    alt.Chart(hitter_compare)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#B31922")
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_on_base:Q", title="Modeled OBP"),
                        tooltip=["scenario:N", alt.Tooltip("sim_on_base:Q", format=".3f"), alt.Tooltip("sim_hr:Q", format=".3f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.dataframe(hitter_compare, use_container_width=True, hide_index=True)
            elif compare_mode == "Pitchers" and "pitchers" in snapshots:
                pitcher_snapshots = snapshots["pitchers"].copy()
                pitcher_names = pitcher_snapshots["player"].drop_duplicates().tolist()
                default_idx = pitcher_names.index("Nathan Eovaldi") if "Nathan Eovaldi" in pitcher_names else 0
                selected_pitcher = st.selectbox("Pitcher", pitcher_names, index=default_idx)
                pitcher_compare = pitcher_snapshots[pitcher_snapshots["player"] == selected_pitcher].copy()
                compare_chart = (
                    alt.Chart(pitcher_compare)
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color="#0D1B33")
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_era:Q", title="Modeled ERA"),
                        tooltip=["scenario:N", alt.Tooltip("sim_era:Q", format=".2f"), alt.Tooltip("delta_era:Q", format=".2f")],
                    )
                    .properties(height=300)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.dataframe(pitcher_compare, use_container_width=True, hide_index=True)
            else:
                st.info("선택한 비교 데이터가 없습니다.")
        else:
            st.info("시뮬레이션 실행 결과는 표시되지만, scenario snapshot 데이터는 아직 불러오지 못했습니다.")