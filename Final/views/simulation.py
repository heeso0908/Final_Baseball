import streamlit as st
import pandas as pd
import altair as alt
from shared import (
    data, RAW_DIR, SIMULATION_OPTIONS, DEFAULT_SIM_RUNS,
    kpi_card, section_badges, finding_box,
    get_simulation_result, get_scenario_snapshots,
    get_simulation_batters, get_simulation_defaults,
    get_live_scenario_results,
    fmt_pct, fmt_num,
)
from agent.tools import list_precomputed_scenarios


_HITTER_MULT_COLUMNS = ["hr_mult", "bb_mult", "k_mult", "single_mult", "double_mult"]
RANGERS_RED = "#B31922"
RANGERS_NAVY = "#0D1B33"
RANGERS_BLUE = "#003278"
CHART_GRAY = "#8F9AAA"

SCENARIO_LABELS = {
    "Baseline 2025": "기준 시뮬 (조정 없음)",
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
        "Pareto": "Pareto",
        "Phase 8": "Phase 8",
        "현재 시뮬": "현재 시뮬",
    }
    return labels.get(str(source), str(source))


def _render_source_legend() -> None:
    st.caption(
        "Manual: 사람이 직접 정한 실행 시나리오 · "
        "Pareto: v5 ML 잔차 모델 6차원 Pareto 후보 · "
        "Phase 8: 12차원 σ NSGA-II 시뮬 직접 평가 (현실 권장 zone σ≤10%)"
    )


def _display_scenario_name(name: str) -> str:
    text = str(name)
    return SCENARIO_LABELS.get(text, text)


def _short_chart_label(name: str) -> str:
    text = _display_scenario_name(name)
    replacements = {
        "Phase 8: 잔차 초과 달성 (σ=7.7%)": "P8\n잔차 초과",
        "Phase 8: 잔차 만회 기준 (σ=8.1%)": "P8\n잔차 만회",
        "Phase 8: 소폭 개선 (σ=6.0%)": "P8\n소폭 개선",
        "공격적 (std=0.652)": "Pareto\n공격적",
        "균형점 (std=0.205)": "Pareto\n균형점",
        "보수적 (std=0.002)": "Pareto\n보수적",
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
    st.markdown("""
    <div class="finding-box" style="margin-top:10px;">
        <strong>통합 Markov 시뮬 기준값</strong> — 타자 Markov + 투수 Markov(Phase 6-7' 메커니즘) 통합 엔진.
        실점 평균 ≈ <b>604</b> (실제 605).
        예상 승수는 실제 81승보다 <b>약 10승 높게</b> 나옵니다.
        이는 시뮬레이션 오류가 아니라 <b>TEX 2025의 Pythagorean 잔차(-9.06승)</b> 때문입니다.
        TEX의 실제 득실 기준 기대 승수는 90.06승이었으나, 세이브 실패·타이밍 불운 등으로 81승에 그쳤습니다.
        시뮬레이션은 득실 기반 기댓값에 수렴하므로 이 잔차는 반영되지 않습니다.
        따라서 절대 승수보다 <b>베이스라인 대비 개선폭(Δ승수)</b>을 기준으로 해석하는 것을 권장합니다.
    </div>
    """, unsafe_allow_html=True)


def _render_custom_controls(selected_scenario: str) -> tuple[dict | None, dict | None, bool]:
    defaults = get_simulation_defaults(str(RAW_DIR))
    scenarios = defaults.get("scenarios", {})
    scenario = scenarios.get(selected_scenario, {})
    stats = dict(scenario.get("stats", {}))
    tex25 = defaults.get("tex25", {})

    if selected_scenario == "Baseline 2025":
        st.info("아무 조건도 변경하지 않은 기준 시뮬레이션입니다. 시나리오 개선폭 비교의 기준점으로 사용합니다.")
        return None, None, True

    if selected_scenario == "Bullpen Upgrade":
        st.markdown("### 불펜 조건")
        sv_baseline = float(tex25.get("sv_pct", stats.get("sv_pct", 0.700)))
        onerun_baseline = float(tex25.get("onerun_wp", stats.get("onerun_wp", 0.500)))
        delta_sv = st.slider(
            "세이브 성공률 변화 (Δ)",
            min_value=-0.300,
            max_value=+0.300,
            value=0.000,
            step=0.001,
            format="%+.3f",
            key="bullpen_sv_pct",
        )
        delta_onerun = st.slider(
            "1점 차 경기 승률 변화 (Δ)",
            min_value=-0.200,
            max_value=+0.200,
            value=0.000,
            step=0.001,
            format="%+.3f",
            key="bullpen_onerun_wp",
        )
        st.caption(
            f"기준값: 세이브 성공률 {sv_baseline:.3f} · 1점 차 승률 {onerun_baseline:.3f} (2025 TEX 실제). "
            "Δ=0이면 기준 시뮬과 동일. 양수=개선, 음수=악화."
        )

        sv_result    = round(sv_baseline + delta_sv, 4)
        onerun_result = round(onerun_baseline + delta_onerun, 4)
        m1, m2 = st.columns(2)
        with m1:
            st.metric(
                "세이브 성공률",
                f"{sv_result:.3f}",
                delta=f"{delta_sv:+.3f}" if abs(delta_sv) >= 1e-9 else "기준값 그대로",
                delta_color="normal" if abs(delta_sv) >= 1e-9 else "off",
            )
        with m2:
            st.metric(
                "1점 차 경기 승률",
                f"{onerun_result:.3f}",
                delta=f"{delta_onerun:+.3f}" if abs(delta_onerun) >= 1e-9 else "기준값 그대로",
                delta_color="normal" if abs(delta_onerun) >= 1e-9 else "off",
            )

        if abs(delta_sv) < 1e-9 and abs(delta_onerun) < 1e-9:
            return None, None, True
        custom_stats_out = {}
        if abs(delta_sv) >= 1e-9:
            custom_stats_out["sv_pct"] = sv_result
        if abs(delta_onerun) >= 1e-9:
            custom_stats_out["onerun_wp"] = onerun_result
        return custom_stats_out, None, True

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
                    format="%.2f",
                    help="0.50 ~ 1.80 사이 값. 1.20이면 홈런 발생을 현재보다 20% 높게 가정합니다.",
                ),
                "bb_mult": st.column_config.NumberColumn(
                    "볼넷 배율",
                    format="%.2f",
                    help="0.50 ~ 1.80 사이 값. 1.20이면 볼넷 발생을 현재보다 20% 높게 가정합니다.",
                ),
                "k_mult": st.column_config.NumberColumn(
                    "삼진 배율",
                    format="%.2f",
                    help="0.50 ~ 1.80 사이 값. 0.90이면 삼진 발생을 현재보다 10% 낮게 가정합니다.",
                ),
                "single_mult": st.column_config.NumberColumn(
                    "단타 배율",
                    format="%.2f",
                    help="0.50 ~ 1.80 사이 값.",
                ),
                "double_mult": st.column_config.NumberColumn(
                    "장타 배율",
                    format="%.2f",
                    help="0.50 ~ 1.80 사이 값.",
                ),
            },
            key="hitter_multiplier_editor",
        )
        st.caption("1.00은 현재 수준입니다. 1.20은 해당 이벤트가 20% 늘어난다는 뜻이고, K 배율은 낮을수록 삼진이 줄어드는 설정입니다.")
        boosts = {
            str(row["player"]): {col: float(min(1.80, max(0.50, row[col]))) for col in _HITTER_MULT_COLUMNS}
            for _, row in edited.iterrows()
        }
        return None, boosts, bool(boosts)

    return None, None, True


def _render_decision_board(
    sim_result_mean: float | None = None,
    sim_label: str | None = None,
    baseline_override: float | None = None,
) -> None:
    st.markdown(
        """
        <div class="glass-card">
            <div class="chart-title">시나리오 의사결정 후보</div>
            <div class="chart-caption">수동 시나리오와 Pareto / Phase 8 후보를 같은 기준으로 비교합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    leaderboard = pd.DataFrame(list_precomputed_scenarios()["scenarios"])

    # Live sim으로 Phase 8 + baseline 재계산 (캐시됨)
    baseline_W = 81.0
    try:
        live = get_live_scenario_results(str(RAW_DIR), n_sims=10)
        if not leaderboard.empty:
            baseline_W = live['baseline_W']
            # baseline row
            mask_base = leaderboard['key'] == 'manual_baseline'
            if mask_base.any():
                leaderboard.loc[mask_base, 'predicted_W'] = baseline_W
                leaderboard.loc[mask_base, 'delta'] = 0.0
            # Phase 8 + Pareto rows — 모두 동일한 baseline_W 기준으로 교체
            for live_key in ['phase8_max', 'phase8_recovery', 'phase8_safe',
                             'pareto_aggressive', 'pareto_balanced', 'pareto_conservative']:
                if live_key not in live:
                    continue
                mask = leaderboard['key'] == live_key
                if mask.any():
                    leaderboard.loc[mask, 'predicted_W'] = live[live_key]['predicted_W']
                    leaderboard.loc[mask, 'delta'] = live[live_key]['delta']
    except Exception:
        pass

    # 사용자가 기준 시뮬을 직접 돌린 경우 → 그 결과를 baseline으로 덮어쓰고 delta 재계산
    if baseline_override is not None:
        baseline_W = round(float(baseline_override), 1)
        mask_base = leaderboard['key'] == 'manual_baseline'
        if mask_base.any():
            leaderboard.loc[mask_base, 'predicted_W'] = baseline_W
            leaderboard.loc[mask_base, 'delta'] = 0.0
        for live_key in ['phase8_max', 'phase8_recovery', 'phase8_safe',
                         'pareto_aggressive', 'pareto_balanced', 'pareto_conservative']:
            mask = leaderboard['key'] == live_key
            if mask.any():
                abs_W = float(leaderboard.loc[mask, 'predicted_W'].iloc[0])
                leaderboard.loc[mask, 'delta'] = round(abs_W - baseline_W, 2)

    # 수동 시뮬 결과를 테이블에 합류
    if sim_result_mean is not None and sim_label is not None:
        manual_delta = round(sim_result_mean - baseline_W, 2)
        manual_row = {
            'key':          'current_sim',
            'source':       '현재 시뮬',
            'label':        sim_label,
            'predicted_W':  round(sim_result_mean, 1),
            'delta':        manual_delta,
            'pred_std':     float('nan'),
            'sigma_norm':   float('nan'),
            'rank':         0,
            'decision_note': '방금 실행한 수동 시뮬레이션 결과',
        }
        leaderboard = pd.concat(
            [pd.DataFrame([manual_row]), leaderboard], ignore_index=True
        )

    if not leaderboard.empty:
        leaderboard["구분_설명"] = leaderboard["source"].map(_scenario_type_label)
        leaderboard["표시_시나리오"] = leaderboard["label"].map(_display_scenario_name)
        show = leaderboard.rename(columns={
            "rank": "순위",
            "구분_설명": "구분",
            "표시_시나리오": "시나리오",
            "delta": "기준 대비 개선승수",
            "predicted_W": "예상 승수",
            "base_predicted_W": "기준 예상 승수",
            "pred_std": "예측 흔들림",
            "sigma_norm": "σ 비용",
            "adjustments_summary": "조정 내역",
            "decision_note": "의사결정 포인트",
        })
        has_base = "기준 예상 승수" in show.columns and show["기준 예상 승수"].notna().any()
        keep = [c for c in ["순위", "구분", "시나리오", "기준 대비 개선승수", "예상 승수", "예측 흔들림", "σ 비용", "조정 내역", "의사결정 포인트"] if c in show.columns]
        show_table = show[keep].copy()
        show_table["기준 대비 개선승수"] = show_table["기준 대비 개선승수"].map(lambda v: f"{float(v):+.3f}")
        show_table["예상 승수"] = show_table["예상 승수"].map(lambda v: f"{float(v):.1f}")
        if "기준 예상 승수" in show_table.columns:
            show_table["기준 예상 승수"] = show_table["기준 예상 승수"].map(
                lambda v: "-" if pd.isna(v) else f"{float(v):.1f}"
            )
        if "예측 흔들림" in show_table.columns:
            show_table["예측 흔들림"] = show_table["예측 흔들림"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.4f}")
        if "σ 비용" in show_table.columns:
            show_table["σ 비용"] = show_table["σ 비용"].map(lambda v: "-" if pd.isna(v) else f"{float(v):.3f}")
        if "조정 내역" in show_table.columns:
            show_table["조정 내역"] = show_table["조정 내역"].fillna("-")
        st.dataframe(
            show_table,
            use_container_width=True,
            hide_index=True,
            height=(len(show_table) + 1) * 35 + 3,
        )
        _render_source_legend()
        st.caption("기준 대비 개선승수: Phase 8 · Pareto · Baseline 모두 동일한 통합 Markov 시뮬 baseline_W 기준으로 계산됩니다. 예측 흔들림은 낮을수록 ML 모델들이 더 비슷하게 본 후보입니다. σ 비용은 Phase 8 전용으로 낮을수록 현실적으로 실행하기 쉬운 정책 조합입니다.")

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
                        domain=["현재 시뮬", "Manual", "Pareto", "Phase 8"],
                        range=["#F5A623", RANGERS_RED, CHART_GRAY, "#2F9E65"],
                    ),
                ),
                tooltip=[
                    alt.Tooltip("구분_설명:N", title="구분"),
                    alt.Tooltip("표시_시나리오:N", title="시나리오"),
                    alt.Tooltip("delta:Q", title="기준 대비 개선승수", format="+.3f"),
                    alt.Tooltip("predicted_W:Q", title="예상 승수", format=".1f"),
                    alt.Tooltip("pred_std:Q", title="예측 흔들림", format=".4f"),
                    alt.Tooltip("sigma_norm:Q", title="σ_norm (정책 변경 비용)", format=".3f"),
                ],
            )
        )
        actual_rule = (
            alt.Chart(pd.DataFrame({"actual_W": [81.0], "label": ["실제 2025 승수 81승"]}))
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



def show():
    st.markdown("""
    <div class="hero-card">
        <span class="pill pill-white">반복 시뮬레이션</span>
        <span class="pill pill-white">잔차 보정 모델</span>
        <span class="pill pill-white">2025 시즌 재구성</span>
        <h1>2025 TEX 시즌 시뮬레이션</h1>
        <p>
        2025 텍사스 레인저스의 실제 시즌을 기준으로 주요 전력 변수 변화가 승수에 미치는 영향을 재구성합니다.
        Baseline 2025를 기준값으로 고정하고, 수동 시나리오와 Pareto / Phase 8 후보를 같은 기준으로 비교합니다.
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
    scenario_keys = list(SIMULATION_OPTIONS)
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

    use_fast_mode = False
    st.caption(
        "통합 시뮬 엔진(integrated_sim) 사용 중 — "
        "타자 Markov(simulator.py) + 투수 Markov(markov_pitching.py) + "
        "Phase 6-7' 메커니즘(high-leverage 패널티, closer 타이밍, 시기별 불펜 풀) 반영. "
        "설정한 반복 횟수만큼 실행되며 완료까지 수 분이 걸릴 수 있습니다."
    )

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
            "상단에서 시나리오와 반복 횟수를 선택한 뒤 시뮬레이션 실행 버튼을 누르면 선택한 조건의 승수 분포와 월별 흐름이 계산됩니다. 아래에서는 Pareto / Phase 8 후보를 먼저 비교할 수 있습니다."
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
    is_integrated = "integrated_n_seasons" in summary

    st.markdown("---")
    st.markdown("## 시뮬레이션 결과 요약")
    if is_integrated:
        n_int = summary.get("integrated_n_seasons", "?")
        st.success(
            f"통합 Markov 엔진 결과 ({n_int}시즌) — "
            "타자 Markov + 투수 Markov(Phase 6-7' 메커니즘 포함)로 계산된 승수 분포입니다. "
            "월별·선수별 세부 데이터는 이 모드에서는 제공되지 않습니다."
        )

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("예상 승수 평균값", fmt_num(summary.get("mean")), f"선택 조건: {_display_scenario_name(st.session_state['sim_scenario'])}", accent="red")
    with k2:
        kpi_card("예상 승수 중앙값", fmt_num(summary.get("median")), f"반복 실행: {st.session_state['sim_runs']}회", accent="navy")
    with k3:
        kpi_card("예상 승수 범위 (P10–P90)", f"{fmt_num(summary.get('p10'))} - {fmt_num(summary.get('p90'))}", f"시뮬레이션의 80% 결과가 이 범위 안에 속합니다", accent="red")
    with k4:
        kpi_card("82승 이상 가능성", fmt_pct(summary.get("over_81_5")), "승률 5할 이상으로 끝날 확률", accent="navy")

    # 기준 시뮬 대비 증감량 — get_live_scenario_results 캐시 재활용 (추가 시뮬 없음)
    cur_scenario = st.session_state.get("sim_scenario", "Baseline 2025")
    if cur_scenario != "Baseline 2025":
        try:
            live_cache  = get_live_scenario_results(str(RAW_DIR), n_sims=10)
            base_mean   = live_cache.get("baseline_W")
            cur_mean    = summary.get("mean")
            cur_over81  = summary.get("over_81_5")
            if base_mean is not None and cur_mean is not None:
                delta_mean = cur_mean - base_mean
                sign  = "▲" if delta_mean >= 0 else "▼"
                color = "#1a7a2e" if delta_mean >= 0 else "#c0392b"
                over81_str = ""
                if cur_over81 is not None:
                    o_sign  = "▲" if cur_over81 >= 0.5 else "▼"
                    o_color = "#1a7a2e" if cur_over81 >= 0.5 else "#c0392b"
                    over81_str = (
                        f"&nbsp;·&nbsp;82승 이상 가능성 "
                        f"<span style='color:{o_color};font-weight:700'>{cur_over81*100:.1f}%</span>"
                    )
                st.markdown(
                    f"""<div style="background:#f0f4ff;border-left:4px solid #3b5bdb;border-radius:6px;
                        padding:12px 16px;margin:12px 0;font-size:14px;line-height:1.8;">
                      <b>기준 시뮬 대비 증감</b>&nbsp;&nbsp;
                      예상 승수 평균
                      <span style='color:{color};font-weight:700;font-size:16px;'>
                        {sign} {abs(delta_mean):.2f}승
                      </span>
                      &nbsp;(기준 {base_mean:.1f}승 → 현재 {cur_mean:.1f}승)
                      {over81_str}
                    </div>""",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    st.markdown("""
    <div class="finding-box">
        <strong>해석 기준</strong><br>
        이 결과는 미래 예측이라기보다 <b>2025 시즌을 조건 변화에 따라 다시 재구성한 결과</b>입니다.
        피타고리안 기대 승수와 실제 승수의 괴리를 설명하기 위해, 시나리오별 승수 분포와 보정값을 함께 확인합니다.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 후보 비교")
    _cur_scenario = st.session_state.get("sim_scenario", "")
    _cur_label    = _display_scenario_name(_cur_scenario) if _cur_scenario else None
    _is_baseline  = _cur_scenario == "Baseline 2025"
    _render_decision_board(
        sim_result_mean=summary.get("mean") if not _is_baseline else None,
        sim_label=_cur_label,
        baseline_override=summary.get("mean") if _is_baseline else None,
    )

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
        st.markdown(f"""
        <div class="glass-card glass-card-accent">
            <div class="kpi-label-custom">선택한 조건</div>
            <div class="kpi-value-custom" style="font-size:22px;">{_display_scenario_name(st.session_state['sim_scenario'])}</div>
            <div class="kpi-sub-custom">같은 조건으로 {st.session_state['sim_runs']}번 시즌을 다시 돌린 결과</div>
            <hr style="margin:14px 0;">
            <div style="font-size:13px; line-height:1.8; color:#344054;">
                <b>88승 이상 가능성</b>: <span class="num">{fmt_pct(summary.get('over_87_5'))}</span><br>
                <b>82승 이상 가능성</b>: <span class="num">{fmt_pct(summary.get('over_81_5'))}</span>
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
                "month": "월",
                "games": "경기",
                "home_games": "홈",
                "away_games": "원정",
                "win_pct": "실제 승률",
                "strength_index": "Strength",
                "difficulty": "Difficulty",
            }
            display = schedule_context.rename(columns=rename_map)
            keep_cols = [col for col in ["월", "경기", "홈", "원정", "실제 승률", "Strength", "Difficulty"] if col in display.columns]
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
            if is_integrated and '선수' in players.columns:
                # 통합 Markov 엔진 결과: 선수별 시뮬 성적 직접 표시
                st.caption("통합 Markov 시뮬레이션 기반 선수별 평균 성적 (N시즌 평균)")
                st.dataframe(players, use_container_width=True, hide_index=True)
            else:
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
            if is_integrated and '투수' in pitchers.columns:
                # 통합 Markov 엔진 결과: 투수별 시뮬 성적 직접 표시
                st.caption("통합 Markov 시뮬레이션 기반 투수별 평균 성적 (N시즌 평균, ERA 오름차순)")
                st.dataframe(pitchers, use_container_width=True, hide_index=True)
            else:
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
        st.markdown("### 시나리오 vs Baseline 선수 성적 비교")
        st.caption("실행한 시나리오의 선수별 시뮬레이션 성적을 Baseline 2025와 비교합니다.")

        cur_result   = st.session_state.get("simulation_result")
        cur_scenario = st.session_state.get("sim_scenario", "Baseline 2025")
        sim_runs_cmp = int(st.session_state.get("sim_runs", DEFAULT_SIM_RUNS))

        if cur_result is None:
            st.info("먼저 상단에서 시뮬레이션을 실행해주세요.")
        elif cur_scenario == "Baseline 2025":
            st.info("Baseline 2025 이외의 시나리오(예: Bullpen Upgrade, Hitter Boost)로 시뮬레이션을 실행하면 Baseline과 비교할 수 있습니다.")
        else:
            try:
                with st.spinner("Baseline 2025 데이터 로딩 중..."):
                    base_result = get_simulation_result(
                        str(RAW_DIR), "Baseline 2025", sim_runs_cmp, fast_mode=False
                    )
            except Exception as exc:
                st.warning(f"Baseline 데이터 로딩 실패: {exc}")
                base_result = None

            if base_result is not None:
                b_base = base_result.get("player_projection")
                p_base = base_result.get("pitcher_projection")
                b_cur  = cur_result.get("player_projection")
                p_cur  = cur_result.get("pitcher_projection")

                compare_mode = st.radio("비교 유형", ["타자", "투수"], horizontal=True)

                if compare_mode == "타자":
                    if b_base is None or b_cur is None or b_base.empty or b_cur.empty:
                        st.info("타자 데이터가 없습니다.")
                    else:
                        b_base_labeled = b_base.copy(); b_base_labeled["시나리오"] = "Baseline 2025"
                        b_cur_labeled  = b_cur.copy();  b_cur_labeled["시나리오"]  = cur_scenario
                        combined = pd.concat([b_base_labeled, b_cur_labeled], ignore_index=True)
                        hitter_names = b_base_labeled["선수"].drop_duplicates().tolist()
                        default_idx = hitter_names.index("Wyatt Langford") if "Wyatt Langford" in hitter_names else 0
                        selected_hitter = st.selectbox("선수 선택", hitter_names, index=default_idx)
                        hitter_compare = combined[combined["선수"] == selected_hitter].copy()
                        compare_chart = (
                            alt.Chart(hitter_compare)
                            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                            .encode(
                                x=alt.X("시나리오:N", title=None),
                                y=alt.Y("OPS:Q", title="OPS", scale=alt.Scale(zero=False)),
                                color=alt.Color(
                                    "시나리오:N",
                                    scale=alt.Scale(
                                        domain=["Baseline 2025", cur_scenario],
                                        range=[CHART_GRAY, RANGERS_RED],
                                    ),
                                    legend=None,
                                ),
                                tooltip=[
                                    alt.Tooltip("시나리오:N",  title="시나리오"),
                                    alt.Tooltip("OPS:Q",      title="OPS",    format=".3f"),
                                    alt.Tooltip("AVG:Q",      title="타율",   format=".3f"),
                                    alt.Tooltip("OBP:Q",      title="출루율", format=".3f"),
                                    alt.Tooltip("SLG:Q",      title="장타율", format=".3f"),
                                    alt.Tooltip("HR/시즌:Q",  title="홈런/시즌", format=".1f"),
                                ],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(compare_chart, use_container_width=True)
                        show_cols = ["시나리오", "PA/시즌", "AVG", "OBP", "SLG", "OPS", "HR/시즌", "BB/시즌", "K/시즌"]
                        show_cols = [c for c in show_cols if c in hitter_compare.columns]
                        st.dataframe(hitter_compare[show_cols], use_container_width=True, hide_index=True)

                else:  # 투수
                    if p_base is None or p_cur is None or p_base.empty or p_cur.empty:
                        st.info("투수 데이터가 없습니다.")
                    else:
                        p_base_labeled = p_base.copy(); p_base_labeled["시나리오"] = "Baseline 2025"
                        p_cur_labeled  = p_cur.copy();  p_cur_labeled["시나리오"]  = cur_scenario
                        combined = pd.concat([p_base_labeled, p_cur_labeled], ignore_index=True)
                        pitcher_names = p_base_labeled["투수"].drop_duplicates().tolist()
                        default_idx = pitcher_names.index("Nathan Eovaldi") if "Nathan Eovaldi" in pitcher_names else 0
                        selected_pitcher = st.selectbox("투수 선택", pitcher_names, index=default_idx)
                        pitcher_compare = combined[combined["투수"] == selected_pitcher].copy()
                        compare_chart = (
                            alt.Chart(pitcher_compare)
                            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
                            .encode(
                                x=alt.X("시나리오:N", title=None),
                                y=alt.Y("ERA:Q", title="ERA", scale=alt.Scale(zero=False)),
                                color=alt.Color(
                                    "시나리오:N",
                                    scale=alt.Scale(
                                        domain=["Baseline 2025", cur_scenario],
                                        range=[CHART_GRAY, RANGERS_NAVY],
                                    ),
                                    legend=None,
                                ),
                                tooltip=[
                                    alt.Tooltip("시나리오:N", title="시나리오"),
                                    alt.Tooltip("ERA:Q",     title="ERA",  format=".2f"),
                                    alt.Tooltip("WHIP:Q",    title="WHIP", format=".2f"),
                                    alt.Tooltip("K%:Q",      title="K%",   format=".3f"),
                                    alt.Tooltip("BB%:Q",     title="BB%",  format=".3f"),
                                    alt.Tooltip("IP/시즌:Q", title="IP/시즌", format=".1f"),
                                ],
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(compare_chart, use_container_width=True)
                        show_cols = ["시나리오", "IP/시즌", "ERA", "WHIP", "K%", "BB%", "HR/시즌", "BB/시즌"]
                        show_cols = [c for c in show_cols if c in pitcher_compare.columns]
                        st.dataframe(pitcher_compare[show_cols], use_container_width=True, hide_index=True)
