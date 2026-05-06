"""TEX 2025 분석 챗봇 — Streamlit + PydanticAI Agent.

실행:
    streamlit run agent/chatbot.py

또는 streamlit_dashboard/app.py에서 import해 페이지로 추가.

환경 변수(.env):
    GEMINI_API_KEY  — Gemini API 키
    GEMINI_MODEL    — 모델명 (기본: gemini-2.0-flash, 안정적인 free tier)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# streamlit run 시 agent/ 만 sys.path에 추가되므로,
# `from agent import tools`가 동작하도록 부모 디렉토리(streamlit_dashboard/) 주입.
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import streamlit as st
from dotenv import load_dotenv
from pydantic_ai import Agent

from agent import tools

load_dotenv(Path(__file__).resolve().parent.parent / '.env')

_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


# ============================================================
# 가상 구독 티어 — 도구 접근 권한만 차등
# ============================================================

TIER_CONFIG: dict[str, dict] = {
    "basic": {
        "label": ' Basic',
        "name": "Basic",
        "tagline": "사전 시나리오·게임로그 조회",
        "tools": {"lookup_pareto", "get_optimization_summary", "query_gamelog", "list_data_sources", "describe_data"},
        "prompt_suffix": (
            "\n\n## 응답 깊이 (Basic 플랜)\n"
            "- 핵심 결과 1-2줄로만 답변합니다.\n"
            "- 한계 설명·부가 해석·교차 비교는 생략합니다.\n"
            "- 도구는 1회 호출로 마무리합니다.\n"
        ),
    },
    "plus": {
        "label": ' Plus',
        "name": "Plus",
        "tagline": "+ 자유 시나리오·팀 비교·시각화",
        "tools": {
            "lookup_pareto", "get_optimization_summary", "query_gamelog",
            "compare_team_2025", "estimate_residual_scenario",
            "plot_scenario_comparison", "plot_team_radar",
            "list_data_sources", "describe_data", "query_data", "plot_custom",
        },
        "prompt_suffix": "",  # 표준 프롬프트 그대로
    },
    "premium": {
        "label": ' Premium',
        "name": "premium",
        "tagline": "+ 이식 시뮬·historical·전체 시각화",
        "tools": {
            "lookup_pareto", "get_optimization_summary", "query_gamelog",
            "compare_team_2025", "estimate_residual_scenario",
            "swap_team_pitching", "query_team_history",
            "plot_scenario_comparison", "plot_historical_distribution",
            "plot_team_radar",
            "list_data_sources", "describe_data", "query_data", "plot_custom",
        },
        "prompt_suffix": (
            "\n\n## 응답 깊이 (Premium 플랜)\n"
            "- 가능하면 도구 2-3개를 조합해 다각도로 답변합니다.\n"
            "- 시뮬 결과는 historical(`query_team_history`) 또는 경쟁팀(`compare_team_2025`)과 "
            "교차 검증 후 제시합니다.\n"
            "- 시나리오 불확실성(`pred_std`)과 한계를 분석가 톤으로 명시합니다.\n"
            "- 사용자가 묻지 않아도 관련된 historical 벤치마크가 있으면 함께 제안합니다.\n"
            "- 사용 불가능한 도구는 정중히 안내합니다.\n"
        ),
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "lookup_pareto": "사전 시나리오 조회",
    "get_optimization_summary": "최적화 요약",
    "query_gamelog": "게임로그 필터 조회",
    "compare_team_2025": "경쟁팀 비교",
    "estimate_residual_scenario": "σ 자유 시뮬",
    "swap_team_pitching": "팀 이식 시뮬",
    "query_team_history": "10년 historical",
    "plot_scenario_comparison": "시나리오 비교 차트",
    "plot_historical_distribution": "historical 분포 차트",
    "plot_team_radar": "팀 비교 레이더 차트",
    "list_data_sources": "데이터셋 목록 조회",
    "describe_data": "데이터셋 컬럼/샘플 확인",
    "query_data": "데이터셋 자유 쿼리 (필터/정렬/제한)",
    "plot_custom": "임의 데이터셋 차트 (bar/line/scatter/hist/box)",
}


def _load_system_prompt() -> str:
    if not _PROMPT_PATH.exists():
        return "당신은 TEX 2025 시즌 분석 보조 에이전트입니다. 한국어로 답변하세요."
    return _PROMPT_PATH.read_text(encoding="utf-8")


def _render_tool_calls(tool_calls: list[dict], key_prefix: str = "") -> None:
    """tool_calls를 UI에 렌더링.

    plotly_figure 타입은 expander 바깥에 차트로 직접 표시,
    나머지는 expander 안 raw text로 표시한다.
    같은 차트가 두 번 그려지지 않도록 key_prefix로 키를 분리한다.
    """
    import json as _json

    # 1) 차트는 바깥에 즉시 표시 (사용자가 바로 보도록)
    for idx, tc in enumerate(tool_calls):
        result = tc.get('result')
        if isinstance(result, dict) and result.get('type') == 'plotly_figure':
            spec = result.get('spec')
            if not spec:
                continue
            try:
                import plotly.graph_objects as _go
                fig = _go.Figure(_json.loads(spec))
                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key=f"chart_{key_prefix}_{idx}",
                )
                if result.get('caption'):
                    st.caption(result['caption'])
            except Exception as e:
                st.warning(f"차트 렌더 실패: {e}")

    # 2) 도구 호출 raw 내역은 expander 안에 (디버깅용)
    with st.expander(f" 도구 호출 내역 ({len(tool_calls)}건)"):
        for tc in tool_calls:
            st.markdown(f"**{tc['name']}**")
            st.code(f"args: {tc['args']}", language="python")
            result = tc.get('result')
            if result is None:
                continue
            # plotly spec은 너무 길어서 메타만 노출
            if isinstance(result, dict) and result.get('type') == 'plotly_figure':
                meta = {k: v for k, v in result.items() if k != 'spec'}
                st.code(f"result (chart meta): {meta}", language="python")
            else:
                st.code(f"result: {result}", language="python")


@st.cache_resource(show_spinner="ML 모델 학습 + Agent 초기화 중...")
def get_agent(tier: str = "basic") -> Agent:
    """Agent 인스턴스를 티어별로 1회씩 생성해 세션 간 공유.

    티어에 따라 등록하는 도구 + 시스템 프롬프트 suffix가 달라진다.
    @st.cache_resource는 tier 인자별로 별도 캐시되므로 전환 시 즉시 반환.
    """
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model_id = f"google-gla:{model_name}"

    config = TIER_CONFIG[tier]
    enabled = config["tools"]
    system_prompt = _load_system_prompt() + config["prompt_suffix"]

    agent = Agent(model_id, system_prompt=system_prompt)

    # 도구 분기 등록 — 티어에 포함된 도구만 활성
    if "estimate_residual_scenario" in enabled:
        @agent.tool_plain
        def estimate_residual_scenario(sigmas: dict[str, float]) -> dict:
            """σ 단위 조정으로 TEX 2025 잔차 점추정 (Monte Carlo 아님).

            4모델 앙상블 평균 + std로 시나리오 결과 + 불확실성을 동시에 산출.
            양수 sigma = 항상 개선 방향 (BB9/HR9/ir_pct 같은 lower-better도 양수=개선).

            주요 출력: predicted_W_calibrated (실제 81승 기준 보정값) 사용 권장.
            predicted_W_raw는 ML 점추정(~86) 그대로의 값.

            조정 가능 피처: sv_pct, SV_pg, onerun_wp, xi_wp, home_away_diff, WHIP,
            k_bb, K9, BB9, HR9, ir_pct, babip_against, go_ao, sb_pct, era_fip_diff.
            """
            return tools.estimate_residual_scenario(sigmas)

    if "lookup_pareto" in enabled:
        @agent.tool_plain
        def lookup_pareto(name: str) -> dict:
            """사전 계산된 v5 통합 비교 시나리오 결과 즉시 반환.

            유효 이름:
            - Pareto: 'aggressive', 'balanced', 'conservative' (또는 한국어 alias)
            - Manual: 'baseline', 'bullpen upgrade', 'hitter' 또는 'hitter boost'
            - Grid: 'best_overall', 'best_bullpen', 'best_closegame',
                    'best_pitching', 'worst_overall', 'baseline'
            - Optimization: 'grid_search', 'nsga2', 'grid_pareto_aggressive'
            """
            return tools.lookup_pareto(name)

    if "get_optimization_summary" in enabled:
        @agent.tool_plain
        def get_optimization_summary() -> dict:
            """노트북 v5 Grid Pareto, NSGA-II 최적화 요약."""
            return tools.get_optimization_summary()

    if "compare_team_2025" in enabled:
        @agent.tool_plain
        def compare_team_2025(team: str) -> dict:
            """TEX 2025와 다른 팀 2025 투수 통계 비교.

            팀 코드(SEA/HOU/ATH/LAA 등) 또는 한국어 별명(매리너스, 애슬래틱스 등) 지원.
            """
            return tools.compare_team_2025(team)

    if "swap_team_pitching" in enabled:
        @agent.tool_plain
        def swap_team_pitching(team: str) -> dict:
            """대상 팀 2025 투수 통계를 TEX 입력에 이식해 시뮬.

            "TEX가 SEA 수준 선발진이었다면 몇 승?" 형태 질문에 답한다.
            K/9, BB/9, HR/9, BABIP를 σ로 변환해 estimate_residual_scenario 호출.
            """
            return tools.swap_team_pitching(team)

    if "query_gamelog" in enabled:
        @agent.tool_plain
        def query_gamelog(
            month: int | None = None,
            opponent: str | None = None,
            home_only: bool = False,
            away_only: bool = False,
            one_run_only: bool = False,
            extra_innings_only: bool = False,
        ) -> dict:
            """TEX 2025 게임로그 필터 조회.

            조건에 맞는 경기들의 승률, 평균 득실 등을 집계. 모든 인자 optional.
            """
            return tools.query_gamelog(
                month=month, opponent=opponent,
                home_only=home_only, away_only=away_only,
                one_run_only=one_run_only, extra_innings_only=extra_innings_only,
            )

    if "query_team_history" in enabled:
        @agent.tool_plain
        def query_team_history(
            year_from: int | None = None,
            year_to: int | None = None,
            residual_min: float | None = None,
            residual_max: float | None = None,
            team: str | None = None,
            top_n: int = 10,
        ) -> dict:
            """10개년 MLB 팀 시즌 historical 조회.

            잔차 범위, 연도 범위, 특정 팀 등으로 필터링. 모든 인자 optional.
            "역대 잔차 -9승 수준 팀이 있었어?" 형태 질문에 답한다.
            """
            return tools.query_team_history(
                year_from=year_from, year_to=year_to,
                residual_min=residual_min, residual_max=residual_max,
                team=team, top_n=top_n,
            )

    if "plot_scenario_comparison" in enabled:
        @agent.tool_plain
        def plot_scenario_comparison(scenarios: list[str]) -> dict:
            """여러 시나리오의 예상 승수를 baseline과 막대그래프로 비교.

            사용자가 "차트로 보여줘", "그래프로", "비교 그래프" 같은 시각화를
            요청하면 호출한다. 이름은 lookup_pareto와 동일.
            예: ['aggressive', 'balanced', 'hopeful', 'best_overall']

            반환 dict의 'spec' 필드는 plotly Figure JSON으로, UI가 직접 렌더한다.
            LLM은 caption + data만 참고하면 충분.
            """
            return tools.plot_scenario_comparison(scenarios)
        
    if "plot_team_radar" in enabled:
        @agent.tool_plain
        def plot_team_radar(teams: list[str]) -> dict:
            """TEX와 1~3개 팀의 투수 지표 레이더 차트 비교.

            축: K/9, BB/9, HR/9, BABIP, GB%, Hard%, SV, BS (8개).
            MLB 30팀 min-max 정규화 + lower-better 자동 반전 → bigger = better.
            "TEX vs 매리너스 차트로 비교", "TEX/SEA/HOU 강약점 한눈에"
            같은 다축 비교 질문에 적합.
            """
            return tools.plot_team_radar(teams)

    if "plot_historical_distribution" in enabled:
        @agent.tool_plain
        def plot_historical_distribution(
            year_from: int | None = None,
            year_to: int | None = None,
            residual_min: float | None = None,
            residual_max: float | None = None,
            highlight_tex_2025: bool = True,
        ) -> dict:
            """historical 잔차 분포 히스토그램. TEX 2025(-9.06)를 점선으로 표시.

            "역대 잔차 분포 보여줘", "TEX 2025가 얼마나 극단적이야?" 류
            위치 비교 질문에 적합. 반환 dict의 'spec'은 UI가 직접 렌더.
            """
            return tools.plot_historical_distribution(
                year_from=year_from, year_to=year_to,
                residual_min=residual_min, residual_max=residual_max,
                highlight_tex_2025=highlight_tex_2025,
            )

    # ─── 데이터 디스커버리 (일반 쿼리) ────────────────────────────────────────
    if "list_data_sources" in enabled:
        @agent.tool_plain
        def list_data_sources() -> dict:
            """사용 가능한 모든 데이터셋의 이름 + 한 줄 설명 + 행/컬럼 수.

            전용 도구(query_gamelog, lookup_pareto 등)로 답이 안 나오는 새로운
            질문이 오면 가장 먼저 이 도구를 호출해 어떤 데이터가 있는지 본다.
            그 다음 describe_data로 컬럼을 확인하고 query_data로 쿼리.
            """
            return tools.list_data_sources()

    if "describe_data" in enabled:
        @agent.tool_plain
        def describe_data(source: str) -> dict:
            """특정 데이터셋의 컬럼(이름·dtype·null·unique·min·max) + 샘플 5행.

            query_data 호출 전에 어떤 컬럼·필터가 가능한지 확인하는 데 쓴다.
            source는 list_data_sources 결과의 'name' 값.
            """
            return tools.describe_data(source)

    if "query_data" in enabled:
        @agent.tool_plain
        def query_data(
            source: str,
            filter: dict | None = None,
            columns: list[str] | None = None,
            sort_by: str | None = None,
            ascending: bool = True,
            limit: int = 50,
        ) -> dict:
            """csv를 필터·선택·정렬해서 row 리스트 반환. 일반 쿼리 도구.

            filter 문법:
              - 등호: {'month': 9, 'opponent': 'SEA'}
              - 범위: {'ERA': {'gte': 4.0, 'lte': 6.0}}
              - 포함: {'name': {'contains': 'Garcia'}}
              - 다중값: {'pos': {'in': ['SP', 'RP']}}
            모든 조건 AND 결합. limit 최대 200.

            전용 도구(lookup_pareto/compare_team_2025 등)로 답이 안 나오는
            조합 질문 ("9월 1점차에서 가장 많이 등판한 투수" 류)에 사용.
            """
            return tools.query_data(
                source=source, filter=filter, columns=columns,
                sort_by=sort_by, ascending=ascending, limit=limit,
            )

    if "plot_custom" in enabled:
        @agent.tool_plain
        def plot_custom(
            source: str,
            chart_type: str,
            x: str,
            y: str | list[str] | None = None,
            color_by: str | None = None,
            filter: dict | None = None,
            sort_by: str | None = None,
            ascending: bool = True,
            limit: int = 500,
            title: str | None = None,
        ) -> dict:
            """DATA_CATALOG의 임의 csv에서 임의 컬럼 조합으로 plotly 차트 생성.

            chart_type: 'bar' | 'line' | 'scatter' | 'histogram' | 'box'.
            전용 plot_* 도구(plot_scenario_comparison, plot_team_radar 등)로
            못 만드는 시각화를 LLM이 직접 조합해서 만들 때 사용.

            예시:
            - 월별 TEX 승률 라인: source='texas_2025_game_log', chart_type='line', x='Date', y='win_pct'
            - 투수별 K9 vs ERA 산점도: source='texas_pitchers_2025', chart_type='scatter', x='K/9', y='ERA'
            - 타자 EV 히스토그램: source='rangers_2025_batters_daily_final', chart_type='histogram', x='EV'

            filter는 query_data와 같은 문법. 행 수가 많은 csv는 limit·filter로 좁혀라.
            """
            return tools.plot_custom(
                source=source, chart_type=chart_type, x=x, y=y,
                color_by=color_by, filter=filter,
                sort_by=sort_by, ascending=ascending, limit=limit, title=title,
            )

    return agent


# ============================================================
# Streamlit UI
# ============================================================

def render():
    """Streamlit 챗봇 페이지 렌더링. app.py에서 호출 가능."""
    # set_page_config는 한 번만 호출 가능 — 이미 호출되었으면 무시
    try:
        st.set_page_config(page_title="TEX 2025 챗봇", page_icon="⚾", layout="wide")
    except st.errors.StreamlitAPIException:
        pass

    st.title("⚾ TEX 2025 분석 챗봇")

    # API 키 확인
    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "환경 변수 `GEMINI_API_KEY`가 설정되어 있지 않습니다. "
            "`Final/.env` 파일을 확인하세요."
        )
        st.stop()

    # 세션 상태 — 대화 메시지 + Agent 메시지 히스토리
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # UI 표시용 [{role, content}]
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = None  # PydanticAI 메시지 히스토리

    # 사이드바 — 구독 플랜 + 컨트롤
    with st.sidebar:
        st.markdown("""
<style>

section[data-testid="stSidebar"] {
    --agent-card-x: 0px;
    --agent-card-gap: 7px;
    --agent-title-gap: 12px;
    --agent-note-gap: 14px;
    --agent-card-min-height: 48px;
    --agent-card-padding-y: 10px;
    --agent-card-padding-x: 12px;
    --agent-card-radius: 10px;
    --agent-card-bg: rgba(255,255,255,0.08);
    --agent-card-border: rgba(255,255,255,0.16);
}

section[data-testid="stSidebar"] .agent-sidebar-list {
    width: calc(100% - (var(--agent-card-x) * 2)) !important;
    margin: var(--agent-title-gap) var(--agent-card-x) 8px var(--agent-card-x) !important;
    padding: 0 !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    gap: var(--agent-card-gap) !important;
}

section[data-testid="stSidebar"] .agent-sidebar-row {
    width: 100% !important;
    min-height: var(--agent-card-min-height) !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
    background: var(--agent-card-bg) !important;
    border: 1px solid var(--agent-card-border) !important;
    border-radius: var(--agent-card-radius) !important;
    padding: var(--agent-card-padding-y) var(--agent-card-padding-x) !important;
    margin: 0 !important;
    color: rgba(255,255,255,0.90) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] .agent-sidebar-icon {
    font-family: "bootstrap-icons" !important;
    color: rgba(255,255,255,0.86) !important;
    font-size: 13px !important;
    line-height: 1 !important;
    flex: 0 0 auto !important;
}

section[data-testid="stSidebar"] .agent-sidebar-desc,
section[data-testid="stSidebar"] .agent-sidebar-code {
    color: rgba(255,255,255,0.88) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
}

section[data-testid="stSidebar"] [data-testid="stVerticalBlock"]:has(.agent-control-zone) {
    gap: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone),
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* chatbot settings: element-container siblings after agent-control-zone marker */
section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container:has([data-testid="stButton"]),
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) ~ [data-testid="stElementContainer"]:has([data-testid="stButton"]) {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] {
    width: calc(100% - (var(--agent-card-x) * 2)) !important;
    margin: 0 var(--agent-card-x) var(--agent-card-gap) var(--agent-card-x) !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) ~ [data-testid="stElementContainer"] [data-testid="stButton"] {
    width: calc(100% - (var(--agent-card-x) * 2)) !important;
    margin: 0 var(--agent-card-x) var(--agent-card-gap) var(--agent-card-x) !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container:has(h3) + .element-container:has([data-testid="stButton"]) [data-testid="stButton"],
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) ~ [data-testid="stElementContainer"]:has(h3) + [data-testid="stElementContainer"]:has([data-testid="stButton"]) [data-testid="stButton"] {
    margin-top: var(--agent-title-gap) !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container:has([data-testid="stButton"]):last-child [data-testid="stButton"],
section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) ~ [data-testid="stElementContainer"]:has([data-testid="stButton"]):last-child [data-testid="stButton"] {
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button {
    width: 100% !important;
    min-height: var(--agent-card-min-height) !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin: 0 !important;
    padding: var(--agent-card-padding-y) var(--agent-card-padding-x) !important;
    background: var(--agent-card-bg) !important;
    border: 1px solid var(--agent-card-border) !important;
    border-radius: var(--agent-card-radius) !important;
    color: rgba(255,255,255,0.90) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    text-align: left !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.agent-control-zone) ~ [data-testid="stElementContainer"] [data-testid="stButton"] > button {
    width: 100% !important;
    min-height: var(--agent-card-min-height) !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    margin: 0 !important;
    padding: var(--agent-card-padding-y) var(--agent-card-padding-x) !important;
    background: var(--agent-card-bg) !important;
    border: 1px solid var(--agent-card-border) !important;
    border-radius: var(--agent-card-radius) !important;
    color: rgba(255,255,255,0.90) !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    text-align: left !important;
    box-shadow: none !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button div {
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button p {
    margin: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    line-height: 1.4 !important;
    text-align: left !important;
    white-space: nowrap !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button:hover {
    background: rgba(255,255,255,0.13) !important;
    border-color: rgba(255,255,255,0.20) !important;
}

section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button:disabled,
section[data-testid="stSidebar"] .element-container:has(.agent-control-zone) ~ .element-container [data-testid="stButton"] > button[disabled] {
    opacity: 1 !important;
    color: rgba(255,255,255,0.90) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.90) !important;
    cursor: default !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio-group"] {
    display: flex !important;
    flex-direction: column !important;
    gap: var(--agent-card-gap) !important;
}

section[data-testid="stSidebar"] [data-baseweb="radio-group"] > * {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] {
    width: calc(100% - (var(--agent-card-x) * 2)) !important;
    margin: var(--agent-title-gap) var(--agent-card-x) 8px var(--agent-card-x) !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] label {
    align-items: center !important;
    min-height: var(--agent-card-min-height) !important;
    padding: var(--agent-card-padding-y) var(--agent-card-padding-x) !important;
    border-radius: var(--agent-card-radius) !important;
    background: var(--agent-card-bg) !important;
    border-color: var(--agent-card-border) !important;
    box-sizing: border-box !important;
    margin: 0 !important;
}

section[data-testid="stSidebar"] .agent-sidebar-note {
    margin: var(--agent-note-gap) var(--agent-card-x) 2px var(--agent-card-x) !important;
}

</style>
""", unsafe_allow_html=True)
        st.markdown("### 구독 플랜")

        tier = st.radio(
            "플랜 선택",
            options=list(TIER_CONFIG.keys()),
            format_func=lambda t: f"{TIER_CONFIG[t]['label']} — {TIER_CONFIG[t]['tagline']}",
            label_visibility="collapsed",
        )

        # 티어 변경 감지 — 대화 초기화 (상위 티어 도구 호출이 하위에서 무의미)
        if st.session_state.get("current_tier") != tier:
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.session_state.current_tier = tier

        st.markdown("---")
        st.markdown('<div class="agent-control-zone" style="display:none">x</div>', unsafe_allow_html=True)
        st.markdown("### 챗봇 설정")
        _model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        st.button(
                f"\uf4f1\u2002Model: {_model}",
                key="agent_model_display",
                use_container_width=True,
                disabled=True,
        )
    
        if st.button(
                "\uf78a\u2002대화 초기화",
                key="agent_reset_btn",
                use_container_width=True,
        ):
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()
    
        if st.button(
                "\uf116\u2002Agent 재로드",
                key="agent_reload_btn",
                use_container_width=True,
        ):
            get_agent.clear()
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()
    
        st.markdown("---")
        st.markdown("### 사용 가능한 도구")
        enabled = TIER_CONFIG[tier]["tools"]
        tool_rows = []
        for name, desc in TOOL_DESCRIPTIONS.items():
            is_enabled = name in enabled
            icon = "\uf26a" if is_enabled else "\uf47a"
            state_class = "" if is_enabled else " locked"
            tool_rows.append(
                f'<div class="agent-sidebar-row{state_class}">'
                f'<span class="agent-sidebar-icon">{icon}</span>'
                f'<span><span class="agent-sidebar-code">{name}</span>'
                f' <span class="agent-sidebar-desc">— {desc}</span></span>'
                f'</div>'
            )
        st.markdown(
            '<div class="agent-sidebar-list">' + "".join(tool_rows) + '</div>',
            unsafe_allow_html=True,
        )
        if tier != "premium":
            locked = [n for n in TOOL_DESCRIPTIONS if n not in enabled]
            if locked:
                st.markdown(
                    f'<div class="agent-sidebar-note">\uf46d 상위 플랜에서 {len(locked)}개 도구 추가 활성화</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("---")
        st.markdown("### 예시 질문")
        example_questions = [
            "균형점 시나리오 결과 알려줘",
            "TEX가 SEA 수준 선발진이었다면?",
            "9월 1점차 경기 승률은?",
            "역대 잔차 -9승 이하 팀 있어?",
            "K9 0.3σ 올리고 BB9 0.4σ 줄이면?",
            "SEA, LAD와 팀 성적 비교 레이더 차트 그려줘",
        ]
        example_rows = "".join(
            f'<div class="agent-sidebar-row">'
            f'<span class="agent-sidebar-icon">\uf4bd</span>'
            f'<span>{question}</span>'
            f'</div>'
            for question in example_questions
        )
        st.markdown(
            f'<div class="agent-sidebar-list">{example_rows}</div>',
            unsafe_allow_html=True,
        )

    # 현재 플랜 + 도구 개수 안내
    st.caption(
        f"현재 플랜: **{TIER_CONFIG[tier]['name']}** · "
        f"활성 도구 {len(enabled)}개 / {len(TOOL_DESCRIPTIONS)}개"
    )

    agent = get_agent(tier)

    # 기존 대화 표시
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                _render_tool_calls(msg["tool_calls"], key_prefix=f"hist_{id(msg)}")

    # 입력
    user_input = st.chat_input("질문을 입력하세요...")
    if not user_input:
        return

    # 사용자 메시지 표시 + 저장
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Agent 호출
    with st.chat_message("assistant"):
        with st.spinner("분석 중..."):
            try:
                result = agent.run_sync(
                    user_input,
                    message_history=st.session_state.agent_history,
                )
            except Exception as e:
                err_text = f" Agent 호출 실패: {e}"
                st.error(err_text)
                # 에러도 채팅 히스토리에 보존 (재실행 시 컨텍스트 유지)
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": err_text,
                    "tool_calls": [],
                })
                return

        st.markdown(result.output)

        # 도구 호출 + 결과 페어로 추출 (LLM 디버깅 가능하도록)
        tool_calls: list[dict] = []
        pending_calls: dict[str, dict] = {}  # tool_call_id → call info
        for msg in result.new_messages():
            for part in getattr(msg, "parts", []):
                cls_name = part.__class__.__name__
                if cls_name == "ToolCallPart":
                    call_info = {
                        "name": getattr(part, "tool_name", "?"),
                        "args": getattr(part, "args", {}),
                        "result": None,
                    }
                    pending_calls[getattr(part, "tool_call_id", id(part))] = call_info
                    tool_calls.append(call_info)
                elif cls_name == "ToolReturnPart":
                    call_id = getattr(part, "tool_call_id", None)
                    if call_id in pending_calls:
                        pending_calls[call_id]["result"] = getattr(part, "content", None)

        if tool_calls:
            _render_tool_calls(tool_calls, key_prefix="latest")

        # 세션 상태 업데이트
        st.session_state.agent_history = result.all_messages()
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result.output,
            "tool_calls": tool_calls,
        })


if __name__ == "__main__":
    render()
