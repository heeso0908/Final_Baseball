import streamlit as st
import pandas as pd
import altair as alt
from shared import (
    data, RAW_DIR, SIMULATION_OPTIONS, DEFAULT_SIM_RUNS,
    kpi_card, page_hero, section_badges, finding_box,
    get_simulation_result, get_scenario_snapshots,
    get_simulation_batters, get_simulation_defaults,
    fmt_pct, fmt_num,
)
from agent.tools import list_precomputed_scenarios


_HR = "<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>"
_HR_SOFT = "<hr style='margin: 24px 0; border:none; border-top:1px solid #EDF0F5;'>"
_HITTER_MULT_COLUMNS = ["hr_mult", "bb_mult", "k_mult", "single_mult", "double_mult"]
RANGERS_RED = "#B31922"
RANGERS_RED_SOFT = "#D04A52"
RANGERS_NAVY = "#0D1B33"
NAVY_SOFT = "#243A5E"
CHART_GRAY = "#B8BDC7"
CHART_MUTED = "#667085"
CHART_GREEN = "#2F9E65"
QUICK_SIM_RUNS = DEFAULT_SIM_RUNS

SCENARIO_LABELS = {
    "Baseline 2025": "기준 시뮬레이션 (조정 없음)",
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
        "NSGA-II": "NSGA-II",
        "현재 시뮬레이션": "현재 시뮬레이션",
    }
    return labels.get(str(source), str(source))


def _render_source_legend() -> None:
    st.markdown(
        "<div style='margin: 2px 0 36px; font-size:12px; color:#94A3B8; line-height:1.65;'>"
        "시나리오 구분 &nbsp;:&nbsp; Manual — 사람이 직접 정한 실행 시나리오 &nbsp;·&nbsp; "
        "NSGA-II — 12차원 σ 다목적 최적화 (Pareto front 50점에서 archetype 3종 도출, 현실 분포 σ ±10~15%)<br>"
        "컬럼 설명 &nbsp;:&nbsp; 기준 대비 개선승수 — 모든 구분이 동일한 통합 Markov 시뮬레이션 기준으로 계산 &nbsp;·&nbsp; "
        "σ 비용 — 낮을수록 현실적으로 실행하기 쉬운 정책 조합 (4-5개 차원 동시 보강의 비현실성은 σ_norm으로 표시)"
        "</div>",
        unsafe_allow_html=True,
    )


def _display_scenario_name(name: str) -> str:
    text = str(name)
    return SCENARIO_LABELS.get(text, text)


def _short_chart_label(name: str) -> str:
    text = _display_scenario_name(name)
    replacements = {
        "NSGA-II: 공격적 (Aggressive, σ=9.2%)": "NSGA\n공격적",
        "NSGA-II: 균형 (Balanced, σ=8.2%)": "NSGA\n균형",
        "NSGA-II: 보수적 (Conservative, σ=3.1%)": "NSGA\n보수적",
    }
    return replacements.get(text, text.replace(" ", "\n"))


def _render_baseline_reference(defaults: dict) -> None:
    tex25 = defaults.get("tex25", {})

    finding_box(
        "시뮬레이션 구조",
        "타자 Markov(PA 단위 상태 전이) + 투수 Markov(Phase 6–7' 이닝별 등판 정책)로 경기당 득실점을 산출하고, "
        "머신러닝 잔차 모델이 세이브 성공률·접전 승률 등 팀 특성에 따른 보정을 추가 적용합니다.<br>"
        "시나리오 평가 시 절대 승수보다 <b>베이스라인 대비 개선폭(Δ승수)</b>을 기준으로 해석하는 것을 권장합니다.",
    )

    st.markdown(_HR_SOFT, unsafe_allow_html=True)
    st.markdown("### TEX 2025 기준값")

    cols = st.columns(4)
    items = [
        ("실제 승수", tex25.get("W"), "2025 최종 결과"),
        ("피타고리안 기대 승수", tex25.get("pyth_W"), "득점/실점만 보면 기대되는 승수"),
        ("세이브 성공률", tex25.get("sv_pct"), "불펜 기준값"),
        ("1점 차 승률", tex25.get("onerun_wp"), "접전 경기 기준값"),
    ]
    for col, (label, value, sub) in zip(cols, items):
        with col:
            digits = 3 if "승률" in label or "성공률" in label else 1
            kpi_card(label, fmt_num(value, digits), sub, accent="navy")

    actual_w  = tex25.get("W",      81.0)
    pyth_w    = tex25.get("pyth_W", 90.0)
    actual_rs = tex25.get("RS")
    actual_ra = tex25.get("RA")
    residual  = actual_w - pyth_w
    res_sign = "+" if residual >= 0 else ""
    rs_ra_plain = (
        f"득점 {fmt_num(actual_rs, 0)} / 실점 {fmt_num(actual_ra, 0)} · "
        if actual_rs is not None and actual_ra is not None else ""
    )
    st.caption(
        f"{rs_ra_plain}피타고리안 기대 승수 {fmt_num(pyth_w, 1)}승 · 실제 {fmt_num(actual_w, 0)}승 · 잔차 {res_sign}{fmt_num(residual, 1)}승 · "
        "잔차(세이브 실패·타이밍 불운)는 머신러닝 모델이 일부 포착하지만 확률적 요인이 남아 시뮬레이션 결과는 피타고리안 기대 승수 수준에 수렴합니다."
    )


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
        st.markdown("### 불펜 강화")
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
            "Δ=0이면 기준 시뮬레이션과 동일. 양수=개선, 음수=악화."
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
        st.markdown("### 타자 강화")
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
            <div class="chart-caption">수동 시나리오와 NSGA-II 다목적 최적화 후보를 같은 기준으로 비교합니다.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    leaderboard = pd.DataFrame(list_precomputed_scenarios()["scenarios"])

    # 후보 보드는 페이지 초기 렌더링을 막지 않도록 저장된 후보값을 우선 사용한다.
    # 현재 Baseline을 직접 실행한 경우에는 아래 baseline_override 블록에서 기준값만 동기화한다.
    baseline_W = 89.80
    nsga_keys = ['nsga_aggressive', 'nsga_balanced', 'nsga_conservative']

    # 사용자가 기준 시뮬레이션을 직접 돌린 경우 → 그 결과를 baseline으로 덮어쓰고 delta 재계산
    if baseline_override is not None:
        baseline_W = round(float(baseline_override), 1)
        mask_base = leaderboard['key'] == 'manual_baseline'
        if mask_base.any():
            leaderboard.loc[mask_base, 'predicted_W'] = baseline_W
            leaderboard.loc[mask_base, 'delta'] = 0.0
        for live_key in nsga_keys:
            mask = leaderboard['key'] == live_key
            if mask.any():
                abs_W = float(leaderboard.loc[mask, 'predicted_W'].iloc[0])
                leaderboard.loc[mask, 'delta'] = round(abs_W - baseline_W, 2)

    # 수동 시뮬레이션 결과를 테이블에 합류
    if sim_result_mean is not None and sim_label is not None:
        manual_delta = round(sim_result_mean - baseline_W, 2)
        manual_row = {
            'key':          'current_sim',
            'source':       '현재 시뮬레이션',
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
                        domain=["현재 시뮬레이션", "Manual", "NSGA-II"],
                        range=[RANGERS_RED_SOFT, CHART_GREEN, NAVY_SOFT],
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
    page_hero(
        "Simulation",
        "2025 TEX 시즌 시뮬레이션",
        "2025 텍사스 레인저스의 실제 시즌을 기준으로 주요 전력 변수 변화가 승수에 미치는 영향을 재구성합니다.<br>"
        "Baseline 2025를 기준값으로 고정하고, 수동 시나리오와 NSGA-II 최적화 후보를 같은 기준으로 비교합니다.",
        [("Monte Carlo", "white"), ("Pythagorean Model", "white"), ("Scenario Compare", "white")],
    )

    st.markdown("## 1. 시나리오 실행")
    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">실행 방식</div>
        <div class="chart-caption">
            시뮬레이션은 페이지 최초 진입 시 자동 실행되지 않습니다.<br>
            기준값을 먼저 확인한 뒤, 아래 조건을 선택하고 버튼을 누르면 경기력·선수 조건 변화가 잔차와 승수에 미치는 결과를 계산합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _render_baseline_reference(get_simulation_defaults(str(RAW_DIR)))

    st.markdown(_HR_SOFT, unsafe_allow_html=True)
    st.markdown("### 조건 선택")
    scenario_keys = list(SIMULATION_OPTIONS)
    scenario_options = [SCENARIO_LABELS.get(s, s) for s in scenario_keys]
    current_scenario_key = st.session_state.get("sim_scenario", scenario_keys[0])
    if current_scenario_key not in scenario_keys:
        current_scenario_key = scenario_keys[0]
    current_scenario_label = SCENARIO_LABELS.get(current_scenario_key, current_scenario_key)
    if st.session_state.get("simulation_scenario_select") not in scenario_options:
        st.session_state["simulation_scenario_select"] = current_scenario_label

    sel_col, run_col = st.columns([2, 1])
    with sel_col:
        selected_scenario_label = st.selectbox(
            "실행할 시나리오",
            scenario_options,
            index=scenario_options.index(current_scenario_label),
            key="simulation_scenario_select",
        )
    with run_col:
        simulation_runs = st.slider(
            "반복 실행 횟수",
            min_value=100,
            max_value=1000,
            value=int(st.session_state.get("sim_runs", QUICK_SIM_RUNS)),
            step=100,
            key="simulation_runs_slider",
        )
    selected_scenario = SCENARIO_KEYS.get(selected_scenario_label, selected_scenario_label)
    custom_stats, custom_boosts, can_run = _render_custom_controls(selected_scenario)

    with st.expander("**엔진 상세 정보**"):
        st.markdown("""
        <div style="font-size:14px; line-height:1.85; color:#334155; padding:10px 6px 24px;">
        통합 시뮬레이션 엔진(integrated_sim) 사용 중 — 타자 Markov(simulator.py) + 투수 Markov(markov_pitching.py) +
        Phase 6-7' 메커니즘(하이 레버리지 패널티, closer 타이밍, 시기별 불펜 풀) 반영<br>
        빠른 대화형 엔진으로 실행됩니다. 반복 횟수가 높을수록 안정적이지만 시간이 더 걸립니다.
        </div>
        """, unsafe_allow_html=True)

    run_click = st.button(
        "시뮬레이션 실행",
        type="primary",
        use_container_width=True,
        disabled=not can_run,
    )

    if "simulation_result" not in st.session_state:
        st.session_state["simulation_result"] = None
        st.session_state["sim_scenario"] = scenario_keys[0]
        st.session_state["sim_runs"] = QUICK_SIM_RUNS

    if st.session_state.get("simulation_result") is None and RAW_DIR.exists():
        try:
            with st.spinner("기준 시뮬레이션 불러오는 중..."):
                st.session_state["simulation_result"] = get_simulation_result(
                    str(RAW_DIR),
                    scenario_keys[0],
                    QUICK_SIM_RUNS,
                    fast_mode=True,
                )
                st.session_state["sim_scenario"] = scenario_keys[0]
                st.session_state["sim_runs"] = QUICK_SIM_RUNS
                st.session_state["sim_custom_stats"] = None
                st.session_state["sim_custom_boosts"] = None
        except Exception as exc:
            st.warning(f"기준 시뮬레이션 자동 로딩 실패: {type(exc).__name__}: {exc}")

    if run_click:
        if not RAW_DIR.exists():
            st.error("data_raw 폴더가 없습니다. app.py/simulator.py에서 쓰던 CSV 파일들을 data_raw 폴더에 넣어주세요.")
            return
        try:
            with st.spinner("시뮬레이션 실행 중..."):
                st.session_state["simulation_result"] = get_simulation_result(
                    str(RAW_DIR),
                    selected_scenario,
                    simulation_runs,
                    custom_stats=custom_stats,
                    custom_boosts=custom_boosts,
                    fast_mode=True,
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
            "상단에서 시나리오와 반복 횟수를 선택한 뒤 시뮬레이션 실행 버튼을 누르면 선택한 조건의 승수 분포와 월별 흐름이 계산됩니다.<br>아래에서는 NSGA-II 후보 archetype을 먼저 비교할 수 있습니다."
        )
        st.markdown(_HR, unsafe_allow_html=True)
        st.markdown("## 2. 후보 비교")
        _render_decision_board()
        return

    summary = result.get("summary", {})
    distribution = result.get("distribution", pd.DataFrame()).copy()
    monthly = result.get("monthly_summary", pd.DataFrame()).copy()
    schedule_context = result.get("schedule_context", pd.DataFrame()).copy()
    players = result.get("player_projection", pd.DataFrame()).copy()
    pitchers = result.get("pitcher_projection", pd.DataFrame()).copy()
    is_integrated = "integrated_n_seasons" in summary

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 2. 시뮬레이션 결과 요약")
    if is_integrated:
        n_int = summary.get("integrated_n_seasons", "?")
        st.success(
            f"통합 Markov 엔진 결과 ({n_int}시즌) — "
            "타자 Markov + 투수 Markov(Phase 6-7' 메커니즘 포함)로 계산된 승수 분포입니다.\n"
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

    rs_mean  = summary.get("rs_mean")
    ra_mean  = summary.get("ra_mean")
    actual_rs = summary.get("actual_rs")
    actual_ra = summary.get("actual_ra")
    actual_w  = summary.get("actual_w")
    if rs_mean is not None and ra_mean is not None:
        r1, r2, r3 = st.columns([1, 1, 2])
        rs_sub = f"실제 {fmt_num(actual_rs, 1)}점" if actual_rs is not None else f"경기당 {fmt_num(rs_mean / 162, 2)}"
        ra_sub = f"실제 {fmt_num(actual_ra, 1)}점" if actual_ra is not None else f"경기당 {fmt_num(ra_mean / 162, 2)}"
        with r1:
            kpi_card("평균 득점 (시즌)", fmt_num(rs_mean, 1), rs_sub, accent="red")
        with r2:
            kpi_card("평균 실점 (시즌)", fmt_num(ra_mean, 1), ra_sub, accent="navy")
        with r3:
            run_diff = rs_mean - ra_mean
            diff_sign = "+" if run_diff >= 0 else ""
            diff_color = "#1a7a2e" if run_diff >= 0 else "#c0392b"
            actual_line = ""
            if actual_rs is not None and actual_ra is not None and actual_w is not None:
                actual_diff = actual_rs - actual_ra
                a_sign = "+" if actual_diff >= 0 else ""
                actual_line = (
                    f"&nbsp;·&nbsp; 실제 득실차 <span style='font-weight:700;'>{a_sign}{fmt_num(actual_diff, 1)}</span>점"
                    f"&nbsp;·&nbsp; 실제 승수 <span style='font-weight:700;'>{fmt_num(actual_w, 0)}</span>승"
                )
            st.markdown(
                f"""<div class="glass-card" style="margin-top:4px;font-size:13px;line-height:1.9;">
                시뮬레이션 득실차&nbsp; <span style="font-weight:700;color:{diff_color};">{diff_sign}{fmt_num(run_diff, 1)}</span>점
                &nbsp;·&nbsp; 피타고리안 기대승수 <span style="font-weight:700;">{fmt_num((rs_mean**1.83) / (rs_mean**1.83 + ra_mean**1.83) * 162, 1)}</span>승
                {actual_line}
                </div>""",
                unsafe_allow_html=True,
            )

    # 기준 시뮬 대비 증감량 — 저장된 기준값으로 즉시 계산해 아래 섹션 렌더링을 막지 않는다.
    cur_scenario = st.session_state.get("sim_scenario", "Baseline 2025")
    if cur_scenario != "Baseline 2025":
        try:
            base_mean   = 89.80
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
                    f"""<div class="finding-box-navy" style="margin:12px 0;">
                      <strong>기준 시뮬레이션 대비 증감</strong>&nbsp;&nbsp;
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

    finding_box(
        "해석 기준",
        "이 결과는 미래 예측이라기보다 <b>2025 시즌을 조건 변화에 따라 다시 재구성한 결과</b>입니다.<br>"
        "피타고리안 기대 승수와 실제 승수의 괴리를 설명하기 위해, 시나리오별 승수 분포와 보정값을 함께 확인합니다.",
    )

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 3. 후보 비교")
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
            st.caption("막대가 높을수록 시뮬레이션에서 자주 나온 승수입니다.\n오른쪽으로 갈수록 더 좋은 시즌 결과입니다.")
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

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 4. 월별 시뮬레이션 흐름")
    monthly_left, monthly_right = st.columns([1.35, 1], gap="large")

    with monthly_left:
        if not monthly.empty and {"month", "mean_wins", "p25_wins", "p75_wins"}.issubset(monthly.columns):
            month_order = list(monthly["month"])
            monthly_band = (
                alt.Chart(monthly)
                .mark_area(opacity=0.25, color=CHART_GRAY)
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

    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 5. 선수별 시나리오 보드")
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
                        str(RAW_DIR), "Baseline 2025", sim_runs_cmp, fast_mode=True
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
                        b_base_labeled = b_base.copy()
                        b_cur_labeled = b_cur.copy()
                        hitter_rename = {
                            "player": "선수",
                            "pa": "PA/시즌",
                            "ops": "OPS",
                            "sim_on_base": "OBP",
                            "sim_hr": "HR/시즌",
                            "sim_k": "K Rate",
                        }
                        b_base_labeled = b_base_labeled.rename(columns=hitter_rename)
                        b_cur_labeled = b_cur_labeled.rename(columns=hitter_rename)
                        b_base_labeled["시나리오"] = "Baseline 2025"
                        b_cur_labeled["시나리오"] = cur_scenario
                        combined = pd.concat([b_base_labeled, b_cur_labeled], ignore_index=True)
                        if "선수" not in combined.columns or "OPS" not in combined.columns:
                            st.info("비교에 필요한 타자 컬럼이 없습니다.")
                            return
                        hitter_names = b_base_labeled["선수"].drop_duplicates().tolist()
                        default_idx = hitter_names.index("Wyatt Langford") if "Wyatt Langford" in hitter_names else 0
                        selected_hitter = st.selectbox("선수 선택", hitter_names, index=default_idx)
                        hitter_compare = combined[combined["선수"] == selected_hitter].copy()
                        hitter_tooltips = [
                            alt.Tooltip("시나리오:N", title="시나리오"),
                            alt.Tooltip("OPS:Q", title="OPS", format=".3f"),
                        ]
                        for col, title, fmt in [
                            ("AVG", "타율", ".3f"),
                            ("OBP", "출루율", ".3f"),
                            ("SLG", "장타율", ".3f"),
                            ("HR/시즌", "홈런/시즌", ".1f"),
                        ]:
                            if col in hitter_compare.columns:
                                hitter_tooltips.append(alt.Tooltip(f"{col}:Q", title=title, format=fmt))
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
                                        range=[CHART_GRAY, RANGERS_RED_SOFT],
                                    ),
                                    legend=None,
                                ),
                                tooltip=hitter_tooltips,
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(compare_chart, use_container_width=True)
                        show_cols = ["시나리오", "선수", "PA/시즌", "AVG", "OBP", "SLG", "OPS", "HR/시즌", "BB/시즌", "K/시즌", "K Rate"]
                        show_cols = [c for c in show_cols if c in hitter_compare.columns]
                        st.dataframe(hitter_compare[show_cols], use_container_width=True, hide_index=True)

                else:  # 투수
                    if p_base is None or p_cur is None or p_base.empty or p_cur.empty:
                        st.info("투수 데이터가 없습니다.")
                    else:
                        p_base_labeled = p_base.copy()
                        p_cur_labeled = p_cur.copy()
                        pitcher_rename = {
                            "player": "투수",
                            "sim_era": "ERA",
                            "sim_whip": "WHIP",
                            "sim_k9": "K/9",
                            "sim_ip": "IP/시즌",
                        }
                        p_base_labeled = p_base_labeled.rename(columns=pitcher_rename)
                        p_cur_labeled = p_cur_labeled.rename(columns=pitcher_rename)
                        p_base_labeled["시나리오"] = "Baseline 2025"
                        p_cur_labeled["시나리오"] = cur_scenario
                        combined = pd.concat([p_base_labeled, p_cur_labeled], ignore_index=True)
                        if "투수" not in combined.columns or "ERA" not in combined.columns:
                            st.info("비교에 필요한 투수 컬럼이 없습니다.")
                            return
                        pitcher_names = p_base_labeled["투수"].drop_duplicates().tolist()
                        default_idx = pitcher_names.index("Nathan Eovaldi") if "Nathan Eovaldi" in pitcher_names else 0
                        selected_pitcher = st.selectbox("투수 선택", pitcher_names, index=default_idx)
                        pitcher_compare = combined[combined["투수"] == selected_pitcher].copy()
                        pitcher_tooltips = [
                            alt.Tooltip("시나리오:N", title="시나리오"),
                            alt.Tooltip("ERA:Q", title="ERA", format=".2f"),
                        ]
                        for col, title, fmt in [
                            ("WHIP", "WHIP", ".2f"),
                            ("K%", "K%", ".3f"),
                            ("BB%", "BB%", ".3f"),
                            ("K/9", "K/9", ".1f"),
                            ("IP/시즌", "IP/시즌", ".1f"),
                        ]:
                            if col in pitcher_compare.columns:
                                pitcher_tooltips.append(alt.Tooltip(f"{col}:Q", title=title, format=fmt))
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
                                        range=[CHART_GRAY, NAVY_SOFT],
                                    ),
                                    legend=None,
                                ),
                                tooltip=pitcher_tooltips,
                            )
                            .properties(height=300)
                        )
                        st.altair_chart(compare_chart, use_container_width=True)
                        show_cols = ["시나리오", "투수", "IP/시즌", "ERA", "WHIP", "K%", "BB%", "K/9", "HR/시즌", "BB/시즌"]
                        show_cols = [c for c in show_cols if c in pitcher_compare.columns]
                        st.dataframe(pitcher_compare[show_cols], use_container_width=True, hide_index=True)
