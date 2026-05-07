import streamlit as st
import pandas as pd
import altair as alt
from shared import (
    data, RAW_DIR, SIMULATION_OPTIONS, DEFAULT_SIM_RUNS,
    kpi_card, section_badges, finding_box,
    get_simulation_result, get_scenario_snapshots,
    get_simulation_batters, get_simulation_defaults,
    fmt_pct, fmt_num,
)
from agent import tools as agent_tools


_HITTER_MULT_COLUMNS = ["hr_mult", "bb_mult", "k_mult", "single_mult", "double_mult"]
RANGERS_RED = "#B31922"
RANGERS_NAVY = "#0D1B33"
RANGERS_BLUE = "#003278"
CHART_GRAY = "#8F9AAA"

SCENARIO_LABELS = {
    "Baseline 2025": "Baseline 2025",
    "Bullpen Upgrade": "불펜 강화",
    "Hitter Boost": "타자 강화",
}
SCENARIO_KEYS = {label: key for key, label in SCENARIO_LABELS.items()}


def _default_hitter_multipliers(player: str) -> dict[str, float]:
    defaults = {key: 1.0 for key in _HITTER_MULT_COLUMNS}
    if player == "Wyatt Langford":
        defaults.update({"hr_mult": 1.30, "bb_mult": 1.15, "k_mult": 0.90, "double_mult": 1.20})
    return defaults


def _scenario_type_label(source: str) -> str:
    labels = {
        "수동": "Manual",
        "그리드": "Grid",
        "Pareto": "Pareto",
    }
    return labels.get(str(source), str(source))


def _render_source_legend() -> None:
    st.caption(
        "Manual: 사람이 직접 정한 실행 시나리오 · "
        "Grid: 여러 조건 조합을 자동으로 훑어본 후보 · "
        "Pareto: 개선 폭과 예측 안정성을 함께 고려한 후보"
    )


def _display_scenario_name(name: str) -> str:
    text = str(name)
    return SCENARIO_LABELS.get(text, text)


def _short_chart_label(name: str) -> str:
    text = _display_scenario_name(name)
    replacements = {
        "공격적 (std=0.652)": "Pareto\n공격적",
        "균형점 (std=0.205)": "Pareto\n균형점",
        "보수적 (std=0.002)": "Pareto\n보수적",
        "전체 복합 개선 상한 (delta=+2.00)": "Grid\n복합 개선",
        "접전 전용 최적 (delta=+1.83)": "Grid\n접전 최적",
        "불펜 전용 최적 (delta=+0.13)": "Grid\n불펜 최적",
        "baseline (현재 수준) (delta=+0.00)": "Grid\n현재 수준",
        "투구 프로파일 최적 (delta=+0.02)": "Grid\n투구 최적",
        "전체 복합 악화 하한 (delta=-0.68)": "Grid\n악화 점검",
    }
    return replacements.get(text, text.replace(" ", "\n"))


def _render_baseline_reference(defaults: dict) -> None:
    tex25 = defaults.get("tex25", {})
    cols = st.columns(4)
    items = [
        ("실제 승수", tex25.get("W"), "2025 최종 결과"),
        ("득실 기반 기대 승수", tex25.get("pyth_W"), "득점/실점만 보면 기대되는 승수"),
        ("세이브 성공률", tex25.get("sv_pct"), "불펜 기준값"),
        ("1점 차 승률", tex25.get("onerun_wp"), "접전 경기 기준값"),
    ]
    for col, (label, value, sub) in zip(cols, items):
        with col:
            digits = 3 if "승률" in label or "성공률" in label else 1
            kpi_card(label, fmt_num(value, digits), sub, accent="navy")
    st.info("Baseline 2025는 실제 기준값 확인용입니다. 시뮬레이션을 돌려도 같은 기준 조건이므로 별도 실행하지 않습니다.")


def _render_custom_controls(selected_scenario: str) -> tuple[dict | None, dict | None, bool]:
    defaults = get_simulation_defaults(str(RAW_DIR))
    scenarios = defaults.get("scenarios", {})
    scenario = scenarios.get(selected_scenario, {})
    stats = dict(scenario.get("stats", {}))

    if selected_scenario == "Bullpen Upgrade":
        st.markdown("### 불펜 조건")
        sv_pct = st.slider(
            "세이브 성공률",
            min_value=0.300,
            max_value=0.950,
            value=float(stats.get("sv_pct", 0.700)),
            step=0.010,
            format="%.3f",
            key="bullpen_sv_pct",
        )
        onerun_wp = st.slider(
            "1점 차 경기 승률",
            min_value=0.200,
            max_value=0.750,
            value=float(stats.get("onerun_wp", 0.500)),
            step=0.010,
            format="%.3f",
            key="bullpen_onerun_wp",
        )
        st.caption("세이브 성공률과 1점 차 경기 승률을 높이면 접전 경기에서 이기는 빈도가 얼마나 달라지는지 확인합니다.")
        return {"sv_pct": sv_pct, "onerun_wp": onerun_wp}, None, True

    if selected_scenario == "Hitter Boost":
        st.markdown("### 타자 조건")
        hitters = get_simulation_batters(str(RAW_DIR))
        default_hitters = ["Wyatt Langford"] if "Wyatt Langford" in hitters else hitters[:1]
        selected_hitters = st.multiselect(
            "조정할 타자",
            hitters,
            default=st.session_state.get("simulation_selected_hitters", default_hitters),
            key="simulation_selected_hitters",
        )
        rows = []
        for player in selected_hitters:
            row = {"player": player}
            row.update(_default_hitter_multipliers(player))
            rows.append(row)

        editor_df = pd.DataFrame(rows, columns=["player", *_HITTER_MULT_COLUMNS])
        edited = st.data_editor(
            editor_df,
            hide_index=True,
            use_container_width=True,
            disabled=["player"],
            column_config={
                "hr_mult": st.column_config.NumberColumn(
                    "홈런 배율",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.05,
                    format="%.2f",
                    help="1.20이면 홈런 발생을 현재보다 20% 높게 가정합니다.",
                ),
                "bb_mult": st.column_config.NumberColumn(
                    "볼넷 배율",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.05,
                    format="%.2f",
                    help="1.20이면 볼넷 발생을 현재보다 20% 높게 가정합니다.",
                ),
                "k_mult": st.column_config.NumberColumn(
                    "삼진 배율",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.05,
                    format="%.2f",
                    help="0.90이면 삼진 발생을 현재보다 10% 낮게 가정합니다.",
                ),
                "single_mult": st.column_config.NumberColumn(
                    "단타 배율",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.05,
                    format="%.2f",
                ),
                "double_mult": st.column_config.NumberColumn(
                    "장타 배율",
                    min_value=0.50,
                    max_value=1.80,
                    step=0.05,
                    format="%.2f",
                ),
            },
            key="hitter_multiplier_editor",
        )
        st.caption("1.00은 현재 수준입니다. 1.20은 해당 이벤트가 20% 늘어난다는 뜻이고, K 배율은 낮을수록 삼진이 줄어드는 설정입니다.")
        boosts = {
            str(row["player"]): {col: float(row[col]) for col in _HITTER_MULT_COLUMNS}
            for _, row in edited.iterrows()
        }
        return None, boosts, bool(boosts)

    return None, None, True


def _render_decision_board() -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="chart-title">시나리오 의사결정 후보</div>
            <div class="chart-caption">
                수동 시나리오와 Grid/Pareto 후보를 같은 기준으로 비교합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    leaderboard = pd.DataFrame(agent_tools.list_precomputed_scenarios()["scenarios"])

    if not leaderboard.empty:
        leaderboard["구분_설명"] = leaderboard["source"].map(_scenario_type_label)
        leaderboard["표시_시나리오"] = leaderboard["label"].map(_display_scenario_name)
        show = leaderboard.rename(columns={
            "rank": "순위",
            "구분_설명": "구분",
            "표시_시나리오": "시나리오",
            "delta": "기준 대비 개선승수",
            "predicted_W": "예상 승수",
            "pred_std": "예측 흔들림",
            "decision_note": "의사결정 포인트",
        })
        keep = [c for c in ["순위", "구분", "시나리오", "기준 대비 개선승수", "예상 승수", "예측 흔들림", "의사결정 포인트"] if c in show.columns]
        show_table = show[keep].copy()
        show_table["기준 대비 개선승수"] = show_table["기준 대비 개선승수"].map(lambda v: f"{float(v):+.3f}")
        show_table["예상 승수"] = show_table["예상 승수"].map(lambda v: f"{float(v):.1f}")
        if "예측 흔들림" in show_table.columns:
            show_table["예측 흔들림"] = show_table["예측 흔들림"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.4f}")
        st.dataframe(
            show_table,
            use_container_width=True,
            hide_index=True,
            height=(len(show_table) + 1) * 35 + 3,
        )
        _render_source_legend()
        st.caption("기준 대비 개선승수는 Baseline 2025와 비교한 예상 승수 개선 폭입니다. 예측 흔들림은 낮을수록 모델들이 더 비슷하게 본 후보입니다.")

        chart_df = leaderboard.copy()
        chart_df["구분_설명"] = chart_df["source"].map(_scenario_type_label)
        chart_df["표시_시나리오"] = chart_df["label"].map(_display_scenario_name)
        chart_df["축_라벨"] = chart_df["label"].map(_short_chart_label)
        bars = (
            alt.Chart(chart_df)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, opacity=0.86)
            .encode(
                x=alt.X(
                    "축_라벨:N",
                    title=None,
                    sort="-y",
                    axis=alt.Axis(labelAngle=0, labelLimit=120, labelOverlap=False),
                ),
                y=alt.Y("predicted_W:Q", title="예상 승수", scale=alt.Scale(zero=False)),
                color=alt.Color(
                    "구분_설명:N",
                    title="구분",
                    scale=alt.Scale(
                        domain=["Manual", "Grid", "Pareto"],
                        range=[RANGERS_RED, RANGERS_BLUE, CHART_GRAY],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("구분_설명:N", title="구분"),
                    alt.Tooltip("표시_시나리오:N", title="시나리오"),
                    alt.Tooltip("delta:Q", title="기준 대비 개선승수", format="+.3f"),
                    alt.Tooltip("predicted_W:Q", title="예상 승수", format=".1f"),
                    alt.Tooltip("pred_std:Q", title="예측 흔들림", format=".4f"),
                ],
            )
        )
        actual_rule = (
            alt.Chart(pd.DataFrame({"actual_W": [81.0], "label": ["실제 승수 81승"]}))
            .mark_rule(color=RANGERS_NAVY, strokeDash=[6, 4], strokeWidth=2)
            .encode(y="actual_W:Q")
        )
        legend_line = (
            alt.Chart(pd.DataFrame({"x": [0], "y": [1], "label": ["실제 2025 승수 81승"]}))
            .mark_rule(color=RANGERS_NAVY, strokeDash=[6, 4], strokeWidth=2)
            .encode(x=alt.value(20), x2=alt.value(58), y=alt.value(18))
        )
        legend_text = (
            alt.Chart(pd.DataFrame({"label": ["실제 2025 승수 81승"]}))
            .mark_text(align="left", baseline="middle", dx=64, dy=18, color=RANGERS_NAVY, fontSize=12, fontWeight="bold")
            .encode(x=alt.value(0), y=alt.value(0), text="label:N")
        )
        chart = (bars + actual_rule + legend_line + legend_text).properties(height=360)
        st.altair_chart(chart, use_container_width=True)

    grid_points = pd.DataFrame(agent_tools.get_grid_pareto_points()["points"])
    if not grid_points.empty:
        st.markdown("### 자동 탐색 대표 후보")
        grid_points["표시_시나리오"] = grid_points["label"].map(_display_scenario_name)
        grid_show = grid_points.rename(columns={
            "표시_시나리오": "시나리오",
            "delta": "기준 대비 개선승수",
            "predicted_W": "예상 승수",
            "pred_std": "예측 흔들림",
            "adjustments_summary": "주요 조정",
        })
        keep = [c for c in ["시나리오", "기준 대비 개선승수", "예상 승수", "예측 흔들림", "주요 조정"] if c in grid_show.columns]
        st.dataframe(grid_show[keep], use_container_width=True, hide_index=True)
        st.caption("자동 탐색 대표 후보는 여러 조합 중 개선 폭과 안정성을 함께 본 선택지입니다. 수동 시나리오를 대체하기보다 비교 기준으로 사용합니다.")


def show():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">반복 시뮬레이션</span>
        <span class="pill pill-white">잔차 보정 모델</span>
        <span class="pill pill-white">2025 시즌 재구성</span>
        <h1>2025 TEX 시즌 시뮬레이션</h1>
        <p>
        2025 텍사스 레인저스의 실제 시즌을 기준으로 주요 전력 변수 변화가 승수에 미치는 영향을 재구성합니다.
        Baseline 2025를 기준값으로 고정하고, 수동 시나리오와 Grid/Pareto 후보를 같은 기준으로 비교합니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    section_badges(
        ("Baseline 2025 / 불펜 강화 / 타자 강화", "red"),
        ("162경기 일정 기반", "navy"),
    )

    st.markdown("---")

    st.markdown("## 시나리오 실행")
    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">실행 방식</div>
        <div class="chart-caption">
            시뮬레이션은 페이지 최초 진입 시 자동 실행되지 않습니다.
            기준값을 먼저 확인한 뒤, 아래 조건을 선택하고 버튼을 누르면 경기력·선수 조건 변화가 잔차와 승수에 미치는 결과를 계산합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Baseline 2025")
    _render_baseline_reference(get_simulation_defaults(str(RAW_DIR)))

    st.markdown("### 조건 선택")
    scenario_keys = [s for s in SIMULATION_OPTIONS if s != "Baseline 2025"]
    scenario_options = [SCENARIO_LABELS.get(s, s) for s in scenario_keys]
    current_scenario_key = st.session_state.get("sim_scenario", scenario_keys[0])
    if current_scenario_key not in scenario_keys:
        current_scenario_key = scenario_keys[0]
    current_scenario_label = SCENARIO_LABELS.get(current_scenario_key, current_scenario_key)
    if st.session_state.get("simulation_scenario_select") not in scenario_options:
        st.session_state["simulation_scenario_select"] = current_scenario_label
    selected_scenario_label = st.selectbox(
        "실행할 시나리오",
        scenario_options,
        index=scenario_options.index(current_scenario_label),
        key="simulation_scenario_select",
    )
    selected_scenario = SCENARIO_KEYS.get(selected_scenario_label, selected_scenario_label)
    simulation_runs = st.slider(
        "반복 실행 횟수",
        min_value=100,
        max_value=1000,
        value=int(st.session_state.get("sim_runs", DEFAULT_SIM_RUNS)),
        step=100,
        key="simulation_runs_slider",
    )
    custom_stats, custom_boosts, can_run = _render_custom_controls(selected_scenario)

    use_fast_mode = not (selected_scenario == "Hitter Boost" and custom_boosts)
    if not use_fast_mode:
        st.info(
            "타자 배율이 설정되어 타석 단위 상세 엔진으로 실행됩니다. "
            "빠른 모드보다 시간이 더 걸릴 수 있습니다."
        )
    else:
        st.caption("기본 엔진은 빠른 경기 단위 Monte Carlo입니다. 기존 타석 단위 상세 엔진보다 훨씬 빠르게 실행됩니다.")

    run_click = st.button(
        "시뮬레이션 실행",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if "simulation_result" not in st.session_state:
        st.session_state["simulation_result"] = None
        st.session_state["sim_scenario"] = scenario_keys[0]
        st.session_state["sim_runs"] = DEFAULT_SIM_RUNS

    if run_click:
        if not RAW_DIR.exists():
            st.error("data_raw 폴더가 없습니다. app.py/simulator.py에서 쓰던 CSV 파일들을 data_raw 폴더에 넣어주세요.")
            return
        try:
            spinner_msg = "시뮬레이션 실행 중 (상세 모드, 시간이 걸릴 수 있습니다)..." if not use_fast_mode else "시뮬레이션 실행 중..."
            with st.spinner(spinner_msg):
                st.session_state["simulation_result"] = get_simulation_result(
                    str(RAW_DIR),
                    selected_scenario,
                    simulation_runs,
                    custom_stats=custom_stats,
                    custom_boosts=custom_boosts,
                    fast_mode=use_fast_mode,
                )
                st.session_state["sim_scenario"] = selected_scenario
                st.session_state["sim_runs"] = simulation_runs
                st.session_state["sim_custom_stats"] = custom_stats
                st.session_state["sim_custom_boosts"] = custom_boosts
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
            "상단에서 시나리오와 반복 횟수를 선택한 뒤 시뮬레이션 실행 버튼을 누르면 선택한 조건의 승수 분포와 월별 흐름이 계산됩니다. 아래에서는 수동 시나리오와 Grid/Pareto 후보를 먼저 비교할 수 있습니다."
        )
        st.markdown("---")
        st.markdown("## 후보 비교")
        _render_decision_board()
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
        kpi_card("예상 승수 평균값", fmt_num(summary.get("mean")), f"선택 조건: {_display_scenario_name(st.session_state['sim_scenario'])}", accent="red")
    with k2:
        kpi_card("예상 승수 중앙값", fmt_num(summary.get("median")), f"반복 실행: {st.session_state['sim_runs']}회", accent="navy")
    with k3:
        kpi_card("예상 승수 범위 (P10–P90)", f"{fmt_num(summary.get('p10'))} - {fmt_num(summary.get('p90'))}", f"시뮬레이션의 80% 결과가 이 범위 안에 속합니다", accent="red")
    with k4:
        kpi_card("82승 이상 가능성", fmt_pct(summary.get("over_81_5")), "승률 5할 이상으로 끝날 확률", accent="navy")

    st.markdown("""
    <div class="finding-box">
        <strong>해석 기준</strong><br>
        이 결과는 미래 예측이라기보다 <b>2025 시즌을 조건 변화에 따라 다시 재구성한 결과</b>입니다.
        피타고리안 기대 승수와 실제 승수의 괴리를 설명하기 위해, 시나리오별 승수 분포와 보정값을 함께 확인합니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 후보 비교")
    _render_decision_board()

    chart_left, chart_right = st.columns([1.35, 1], gap="large")

    with chart_left:
        st.markdown("### 승수 분포")
        if not distribution.empty and "wins" in distribution.columns:
            win_chart = (
                alt.Chart(distribution)
                .mark_bar(color=RANGERS_RED, opacity=0.84, binSpacing=2)
                .encode(
                    x=alt.X("wins:Q", bin=alt.Bin(maxbins=18), title="예상 승수"),
                    y=alt.Y("count():Q", title="나온 횟수"),
                    tooltip=[alt.Tooltip("count():Q", title="나온 횟수")],
                )
                .properties(height=330)
            )
            st.altair_chart(win_chart, use_container_width=True)
            st.caption("막대가 높을수록 시뮬레이션에서 자주 나온 승수입니다. 오른쪽으로 갈수록 더 좋은 시즌 결과입니다.")
        else:
            st.info("승수 분포 데이터가 없습니다.")

    with chart_right:
        st.markdown("### 결과 읽는 법")
        residual_bonus = float(summary.get("residual_bonus", 0.0) or 0.0)
        calibration_offset = float(summary.get("calibration_offset", 0.0) or 0.0)
        model_note = (
            "현재 빠른 실행 모드는 경기 일정과 선택 조건을 중심으로 계산하며, 별도의 ML 잔차 보정을 더하지 않았습니다."
            if abs(residual_bonus) < 1e-9 and abs(calibration_offset) < 1e-9
            else "선택 조건에 ML 잔차 보정값을 더해 승수 분포를 조정했습니다."
        )
        st.markdown(f"""
        <div class="glass-card glass-card-accent">
            <div class="kpi-label-custom">선택한 조건</div>
            <div class="kpi-value-custom" style="font-size:22px;">{_display_scenario_name(st.session_state['sim_scenario'])}</div>
            <div class="kpi-sub-custom">같은 조건으로 {st.session_state['sim_runs']}번 시즌을 다시 돌린 결과</div>
            <hr style="margin:14px 0;">
            <div style="font-size:13px; line-height:1.8; color:#344054;">
                <b>88승 이상 가능성</b>: <span class="num">{fmt_pct(summary.get('over_87_5'))}</span><br>
                <b>82승 이상 가능성</b>: <span class="num">{fmt_pct(summary.get('over_81_5'))}</span><br>
                <b>시나리오 보정</b>: <span class="num">{fmt_num(residual_bonus + calibration_offset, 2, sign=True)}</span>승<br>
                <span style="color:#667085;">{model_note}</span>
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
                .mark_area(opacity=0.20, color=CHART_GRAY)
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("p25_wins:Q", title="월별 예상 승수"),
                    y2="p75_wins:Q",
                    tooltip=[
                        alt.Tooltip("month:N", title="월"),
                        alt.Tooltip("mean_wins:Q", title="평균", format=".2f"),
                        alt.Tooltip("p25_wins:Q", title="낮은 쪽 범위", format=".2f"),
                        alt.Tooltip("p75_wins:Q", title="높은 쪽 범위", format=".2f"),
                    ],
                )
            )
            monthly_line = (
                alt.Chart(monthly)
                .mark_line(point=True, strokeWidth=3, color=RANGERS_NAVY)
                .encode(
                    x=alt.X("month:N", sort=month_order, title=None),
                    y=alt.Y("mean_wins:Q", title="월별 예상 승수"),
                )
            )
            st.altair_chart((monthly_band + monthly_line).properties(height=300), use_container_width=True)
            st.caption("남색 선은 월별 평균, 회색 영역은 흔히 나오는 범위입니다.")
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
    hitter_tab, pitcher_tab, scenario_tab = st.tabs(
        ["Hitters", "Pitchers", "Scenario Board"]
    )

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
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=RANGERS_RED)
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_on_base:Q", title="출루 능력 지표"),
                        tooltip=[
                            alt.Tooltip("scenario:N", title="시나리오"),
                            alt.Tooltip("sim_on_base:Q", title="출루 능력", format=".3f"),
                            alt.Tooltip("sim_hr:Q", title="홈런 비율", format=".3f"),
                        ],
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
                    .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5, color=RANGERS_NAVY)
                    .encode(
                        x=alt.X("scenario:N", title=None),
                        y=alt.Y("sim_era:Q", title="예상 ERA"),
                        tooltip=[
                            alt.Tooltip("scenario:N", title="시나리오"),
                            alt.Tooltip("sim_era:Q", title="예상 ERA", format=".2f"),
                            alt.Tooltip("delta_era:Q", title="ERA 변화", format=".2f"),
                        ],
                    )
                    .properties(height=300)
                )
                st.altair_chart(compare_chart, use_container_width=True)
                st.dataframe(pitcher_compare, use_container_width=True, hide_index=True)
            else:
                st.info("선택한 비교 데이터가 없습니다.")
        else:
            st.info("시뮬레이션 실행 결과는 표시되지만, scenario snapshot 데이터는 아직 불러오지 못했습니다.")
