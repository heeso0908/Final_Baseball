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
import json
from pathlib import Path

# streamlit run 시 agent/ 만 sys.path에 추가되므로,
# `from agent import tools`가 동작하도록 부모 디렉토리(streamlit_dashboard/) 주입.
_PARENT = str(Path(__file__).resolve().parent.parent)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import streamlit as st
from dotenv import load_dotenv
from google.oauth2 import service_account
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent import tools

# .env는 워크스페이스 루트(Final_Baseball/)에 있음
# Final/streamlit/agent/chatbot.py → ../../../ = Final_Baseball/
load_dotenv(Path(__file__).resolve().parent.parent.parent.parent / '.env')

_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


def get_secret(key: str, default=None):
    """Streamlit Cloud Secrets 우선, 없으면 로컬 .env 사용."""
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)


GCP_PROJECT_ID = get_secret("GCP_PROJECT_ID")
GCP_LOCATION = get_secret("GCP_LOCATION", "global")
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.5-flash")
GCP_SERVICE_ACCOUNT_JSON = get_secret("GCP_SERVICE_ACCOUNT_JSON")
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")


def create_vertex_model() -> GoogleModel:
    """Streamlit Cloud Secrets의 서비스 계정 JSON으로 Vertex AI Gemini 모델 생성."""

    if not GCP_PROJECT_ID:
        raise RuntimeError(
            "GCP_PROJECT_ID가 설정되어 있지 않습니다. "
            "Streamlit Secrets 또는 .env를 확인하세요."
        )

    if not GCP_SERVICE_ACCOUNT_JSON:
        raise RuntimeError(
            "GCP_SERVICE_ACCOUNT_JSON이 설정되어 있지 않습니다. "
            "Streamlit Cloud Secrets에 서비스 계정 JSON을 등록하세요."
        )

    service_account_info = json.loads(GCP_SERVICE_ACCOUNT_JSON)

    private_key = service_account_info.get("private_key", "")

    # Streamlit Secrets 복사 과정에서 \n 처리 방식이 달라지는 경우 보정
    private_key = private_key.replace("\\n", "\n")
    service_account_info["private_key"] = private_key.strip() + "\n"

    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )

    provider = GoogleProvider(
        vertexai=True,
        project=GCP_PROJECT_ID,
        location=GCP_LOCATION,
        credentials=credentials,
    )

    return GoogleModel(GEMINI_MODEL, provider=provider)


def create_model() -> GoogleModel:
    """GEMINI_API_KEY 우선, 없으면 Vertex AI 사용."""
    if GEMINI_API_KEY:
        provider = GoogleProvider(api_key=GEMINI_API_KEY)
        return GoogleModel(GEMINI_MODEL, provider=provider)
    return create_vertex_model()

# ============================================================
# 가상 구독 티어 — 도구 접근 권한만 차등
# ============================================================

TIER_CONFIG: dict[str, dict] = {
    "basic": {
        "icon": "lightning-charge-fill",
        "icon_glyph": "\uf46c",
        "label": "Basic",
        "name": "Basic",
        "tagline": "사전 시나리오 · 게임로그 · 선수 성적 조회",
        "tools": {"lookup_pareto", "get_optimization_summary", "query_gamelog", "list_data_sources", "describe_data", "get_player_stats"},
        "prompt_suffix": (
            "\n\n## 응답 깊이 (Basic 플랜)\n"
            "- 핵심 결과 1-2줄로만 답변합니다.\n"
            "- 한계 설명·부가 해석·교차 비교는 생략합니다.\n"
            "- 도구는 1회 호출로 마무리합니다.\n"
        ),
    },
    "plus": {
        "icon": "gem",
        "icon_glyph": "\uf3e6",
        "label": "Plus",
        "name": "Plus",
        "tagline": "+ 자유 시나리오 · 팀 비교 · 시각화",
        "tools": {
            "lookup_pareto", "get_optimization_summary", "query_gamelog",
            "compare_team_2025", "estimate_residual_scenario",
            "plot_scenario_comparison", "plot_team_radar",
            "list_data_sources", "describe_data", "query_data", "plot_custom",
            "get_player_stats", "simulation_player_breakdown",
        },
        "prompt_suffix": "",  # 표준 프롬프트 그대로
    },
    "premium": {
        "icon": "stars",
        "icon_glyph": "\uf589",
        "label": "Premium",
        "name": "Premium",
        "tagline": "+ 이식 시뮬 · historical · 전체 시각화",
        "tools": {
            "lookup_pareto", "get_optimization_summary", "query_gamelog",
            "compare_team_2025", "estimate_residual_scenario",
            "swap_team_pitching", "query_team_history",
            "plot_scenario_comparison", "plot_historical_distribution",
            "plot_team_radar",
            "list_data_sources", "describe_data", "query_data", "plot_custom",
            "get_player_stats", "simulation_player_breakdown",
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
    "list_data_sources": "데이터셋 목록 조회",
    "describe_data": "데이터셋 컬럼/샘플 확인",
    "get_player_stats": "선수 성적 조회 (투수·타자)",
    "compare_team_2025": "경쟁팀 비교",
    "estimate_residual_scenario": "σ 자유 시나리오",
    "plot_scenario_comparison": "시나리오 비교 차트",
    "plot_team_radar": "팀 비교 레이더 차트",
    "query_data": "데이터셋 자유 쿼리 (필터/정렬/제한)",
    "plot_custom": "임의 데이터셋 차트 (bar/line/scatter/hist/box)",
    "simulation_player_breakdown": "시뮬레이션 → 선수별 기여 분석",
    "swap_team_pitching": "팀 이식 시뮬레이션",
    "query_team_history": "10년 historical",
    "plot_historical_distribution": "historical 분포 차트",
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
    with st.expander(f"🧰 도구 호출 내역 ({len(tool_calls)}건)"):
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


@st.cache_resource(show_spinner="머신러닝 모델 학습 + Agent 초기화 중...")
def get_agent(tier: str = "basic") -> Agent:
    """Agent 인스턴스를 티어별로 1회씩 생성해 세션 간 공유.

    티어에 따라 등록하는 도구 + 시스템 프롬프트 suffix가 달라진다.
    @st.cache_resource는 tier 인자별로 별도 캐시되므로 전환 시 즉시 반환.
    """
    model = create_model()

    config = TIER_CONFIG[tier]
    enabled = config["tools"]
    system_prompt = _load_system_prompt() + config["prompt_suffix"]

    agent = Agent(model, system_prompt=system_prompt)

    # 도구 분기 등록 — 티어에 포함된 도구만 활성
    if "estimate_residual_scenario" in enabled:
        @agent.tool_plain
        def estimate_residual_scenario(sigmas: dict[str, float]) -> dict:
            """σ 단위 조정으로 TEX 2025 잔차 점추정 (Monte Carlo 아님).

            4모델 앙상블 평균 + std로 시나리오 결과 + 불확실성을 동시에 산출.
            양수 sigma = 항상 개선 방향 (BB9/HR9/ir_pct 같은 lower-better도 양수=개선).

            주요 출력: predicted_W_calibrated (실제 81승 기준 보정값) 사용 권장.
            predicted_W_raw는 머신러닝 점추정(~86) 그대로의 값.

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
            - Phase 8: 'phase8_max', 'phase8_recovery', 'phase8_safe'
            - Manual: 'baseline', 'bullpen upgrade', 'hitter' 또는 'hitter boost'
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

    if "get_player_stats" in enabled:
        @agent.tool_plain
        def get_player_stats(name: str) -> dict:
            """선수 이름으로 TEX 2025 주요 성적을 한 번에 조회.

            투수·타자 구분 없이 이름(부분 일치 가능)으로 검색한다.
            "Garcia 성적 알려줘", "Seager wRC+ 얼마야?", "Leiter ERA?"
            같은 선수별 성적 질문에 가장 먼저 이 도구를 사용한다.

            투수: ERA / FIP / K/9 / BB/9 / HR/9 / BABIP / GB% / WAR / SV / BS
                  + WPA / Clutch / pLI + 경기별 요약
            타자: wRC+ / wOBA / xwOBA / EV / Barrel% / HR + WPA / Clutch

            Args:
                name: 선수 이름 또는 성 (예: 'Garcia', 'Seager', 'Leiter').
            """
            return tools.get_player_stats(name)

    if "simulation_player_breakdown" in enabled:
        @agent.tool_plain
        def simulation_player_breakdown(sigmas: dict[str, float]) -> dict:
            """시뮬 σ 조정이 어떤 선수 성적과 연결되는지 분석.

            estimate_residual_scenario와 동일한 sigmas dict를 받아서,
            각 피처 개선 목표를 달성하려면 현재 TEX 투수진 중 누가 얼마나 바뀌어야
            하는지를 선수 단위로 구체화한다.

            "BB/9 0.5σ 개선이 실제로 어떤 투수 이야기야?",
            "K9 올리려면 누가 얼마나 늘려야 해?" 같은 질문에 사용.

            매핑 가능한 피처: K9, BB9, HR9, babip_against, WHIP, go_ao.
            sv_pct / onerun_wp 등 팀 집계 피처는 별도 설명으로 반환.

            Args:
                sigmas: {'K9': 0.3, 'BB9': 0.4} 형태. 양수 = 개선.
            """
            return tools.simulation_player_breakdown(sigmas)

    return agent


# ============================================================
# Streamlit UI
# ============================================================

_PANEL_CSS = """<style>
@import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");

/* AI Agent left panel */
[data-testid="column"]:has(.tex-agent-panel-marker),
[data-testid="stColumn"]:has(.tex-agent-panel-marker),
[data-tex-panel="1"] {
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(179,25,34,0.10) 0%, transparent 54%),
        linear-gradient(180deg, #0D1B33 0%, #071A35 100%) !important;
    border-radius: 18px !important;
    padding: 0 !important;
    min-height: 200px;
    box-shadow: 0 18px 42px -24px rgba(13,27,51,0.55);
    overflow: hidden !important;
}

[data-testid="column"]:has(.tex-agent-panel-marker) > div[data-testid="stVerticalBlock"],
[data-testid="stColumn"]:has(.tex-agent-panel-marker) > div[data-testid="stVerticalBlock"],
[data-tex-panel="1"] > div[data-testid="stVerticalBlock"] {
    padding: 8px 0 28px 0 !important;
    gap: 0 !important;
}

.tex-agent-panel-marker {
    display: none !important;
}

/* Section labels */
.ap-label {
    display: block !important;
    color: rgba(255,255,255,0.86) !important;
    font-family: "Sora", "Manrope", "Noto Sans KR", sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 900 !important;
    letter-spacing: 0.04em !important;
    text-transform: none !important;

    margin: 32px 16px 16px 16px !important;
    padding: 0 !important;
    line-height: 1.25 !important;

    position: relative !important;
    z-index: 5 !important;
}

/* 구분선 */
.ap-hr {
    height: 1px;
    background: rgba(255,255,255,0.105);
    margin: 12px 20px 8px 20px !important;
}

/* Card list */
.ap-list {
    width: calc(100% - 22px);
    margin: 0 11px;
    display: flex;
    flex-direction: column;
    gap: 9px;
    box-sizing: border-box;
}

/* Card row */
.ap-row {
    width: 100%;
    display: flex;
    align-items: flex-start;
    gap: 10px;
    box-sizing: border-box;

    background: rgba(255,255,255,0.065);
    border: 1px solid rgba(255,255,255,0.115);
    border-radius: 13px;

    padding: 10px 11px;
    color: rgba(255,255,255,0.88);

    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif;
    font-size: 11.6px;
    font-weight: 600;
    line-height: 1.45;

    overflow: hidden;
}

.ap-row.locked {
    background: rgba(255,255,255,0.035);
    border-color: rgba(255,255,255,0.075);
    color: rgba(255,255,255,0.50);
}

.ap-icon {
    flex: 0 0 18px;
    width: 18px;
    height: 18px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    border-radius: 999px;
    background: rgba(255,255,255,0.12);

    font-family: "bootstrap-icons" !important;
    font-size: 10px;
    color: rgba(255,255,255,0.82);
    line-height: 1;

    margin-top: 1px;
}

.ap-icon .bi {
    font-family: "bootstrap-icons" !important;
    font-size: 10px !important;
    line-height: 1 !important;
}

.ap-row.locked .ap-icon {
    background: rgba(255,255,255,0.06);
    color: rgba(255,255,255,0.42);
}

.ap-body {
    min-width: 0;
    flex: 1 1 auto;
    display: flex;
    flex-direction: column;
    gap: 3px;
}

.ap-name {
    display: block;
    color: rgba(255,255,255,0.96);
    font-family: "JetBrains Mono", "Manrope", monospace;
    font-size: 11.2px;
    font-weight: 800;
    line-height: 1.35;

    white-space: normal;
    word-break: break-word;
    overflow-wrap: anywhere;
}

.ap-desc {
    display: block;
    color: rgba(255,255,255,0.62);
    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif;
    font-size: 11.2px;
    font-weight: 500;
    line-height: 1.42;

    white-space: normal;
    word-break: keep-all;
    overflow-wrap: break-word;
}

.ap-example {
    display: block;
    color: rgba(255,255,255,0.84);
    font-size: 11.5px;
    font-weight: 600;
    line-height: 1.45;
    word-break: keep-all;
}

/* 상위 플랜 안내 영역 */
.ap-note-wrap {
    width: calc(100% - 22px);
    margin: 14px 11px 26px 11px !important;
    box-sizing: border-box;
}

/* 상위 플랜 안내 문구 */
.ap-note {
    width: 100%;
    box-sizing: border-box;

    display: flex;
    align-items: center;
    gap: 7px;

    padding: 10px 12px !important;
    margin: 0 !important;

    border-radius: 12px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.11);

    color: rgba(255,255,255,0.74);
    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif;
    font-size: 11.5px !important;
    font-weight: 650;
    line-height: 1.45 !important;

    word-break: keep-all;
}

.ap-note .bi {
    font-family: "bootstrap-icons" !important;
    font-size: 11px !important;
    line-height: 1 !important;
    color: rgba(255,255,255,0.72);
}

/* 제목 바로 아래 리스트 간격 */
.ap-label + .ap-list {
    margin-top: 0 !important;
}

/* Section title spacing */
.ap-label {
    display: block !important;
    color: rgba(255,255,255,0.86) !important;
    font-family: "Sora", "Manrope", "Noto Sans KR", sans-serif !important;
    font-size: 13.5px !important;
    font-weight: 900 !important;
    letter-spacing: 0.03em !important;
    text-transform: none !important;

    margin: 30px 16px 0 16px !important;
    padding: 0 !important;
    line-height: 1.25 !important;

    position: relative !important;
    z-index: 5 !important;
}

/* 제목과 첫 번째 박스 사이 간격 */
.ap-title-gap {
    height: 30px;
}

/* 첫 번째 섹션은 패널 상단과 너무 멀어지지 않게 */
[data-tex-panel="1"] .ap-label:first-of-type,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-label:first-of-type {
    margin-top: 28px !important;
}

/* 구분선 다음 제목 간격 */
.ap-hr + div .ap-label,
.ap-hr + .ap-label {
    margin-top: 26px !important;
}

/* Buttons */
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton,
[data-tex-panel="1"] .stButton {
    width: calc(100% - 22px) !important;
    margin: 0 11px 9px 11px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button,
[data-tex-panel="1"] .stButton > button {
    width: 100% !important;
    min-height: 42px !important;
    box-sizing: border-box !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 10px 11px !important;
    margin: 0 !important;

    background: rgba(255,255,255,0.065) !important;
    border: 1px solid rgba(255,255,255,0.115) !important;
    border-radius: 13px !important;

    color: rgba(255,255,255,0.86) !important;

    font-family: "bootstrap-icons", "Manrope", "Noto Sans KR", system-ui, sans-serif !important;
    font-size: 11.8px !important;
    font-weight: 650 !important;
    text-align: left !important;
    line-height: 1.35 !important;

    box-shadow: none !important;
    opacity: 1 !important;
}

[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button:hover,
[data-tex-panel="1"] .stButton > button:hover {
    background: rgba(255,255,255,0.105) !important;
    border-color: rgba(255,255,255,0.18) !important;
}

[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button div,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button p,
[data-tex-panel="1"] .stButton > button div,
[data-tex-panel="1"] .stButton > button p {
    color: inherit !important;
    font-family: "bootstrap-icons", "Manrope", "Noto Sans KR", system-ui, sans-serif !important;
    font-size: inherit !important;
    font-weight: inherit !important;

    margin: 0 !important;
    padding: 0 !important;

    line-height: 1.35 !important;
    white-space: normal !important;
    overflow-wrap: anywhere !important;
    text-align: left !important;
}

[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button:disabled,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button[disabled],
[data-tex-panel="1"] .stButton > button:disabled,
[data-tex-panel="1"] .stButton > button[disabled] {
    opacity: 1 !important;
    color: rgba(255,255,255,0.50) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.50) !important;
    cursor: default !important;
    background: rgba(255,255,255,0.035) !important;
    border-color: rgba(255,255,255,0.06) !important;
}

/* 패널 첫 번째 제목: 구독 플랜 */
[data-tex-panel="1"] .ap-label:first-of-type,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-label:first-of-type {
    margin-top: 26px !important;
}

/* 상위 플랜 안내 박스 전후 강제 여백 */
.ap-note-spacer-top {
    height: 25px;
}

.ap-note-spacer-bottom {
    height: 26px;
}

.ap-note-wrap {
    width: calc(100% - 22px);
    margin: 0 11px !important;
    box-sizing: border-box;
}

.ap-note {
    width: 100%;
    box-sizing: border-box;

    display: flex;
    align-items: center;
    gap: 7px;

    padding: 10px 12px !important;
    margin: 0 !important;

    border-radius: 12px;
    background: rgba(255,255,255,0.055);
    border: 1px solid rgba(255,255,255,0.11);

    color: rgba(255,255,255,0.74);
    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif;
    font-size: 11.5px !important;
    font-weight: 650;
    line-height: 1.45 !important;
}

.ap-note .bi {
    font-family: "bootstrap-icons" !important;
    font-size: 11px !important;
    line-height: 1 !important;
    color: rgba(255,255,255,0.72);
}

/* 전체 카드 폭 통일: 플랜 / 버튼 / 도구 / 예시 질문 */
[data-tex-panel="1"] .plan-list,
[data-testid="column"]:has(.tex-agent-panel-marker) .plan-list,
[data-tex-panel="1"] .ap-list,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-list,
[data-tex-panel="1"] .ap-note-wrap,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-note-wrap,
[data-tex-panel="1"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton {
    width: calc(100% - 28px) !important;
    max-width: calc(100% - 28px) !important;
    margin-left: 14px !important;
    margin-right: 14px !important;
    box-sizing: border-box !important;
}

/* 플랜 카드 내부 */
[data-tex-panel="1"] .plan-card,
[data-testid="column"]:has(.tex-agent-panel-marker) .plan-card {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* 도구/예시 질문 카드 내부 */
[data-tex-panel="1"] .ap-row,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-row {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* 챗봇 설정 버튼 내부 */
[data-tex-panel="1"] .stButton > button,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* 각 리스트의 아래 여백은 유지 */
[data-tex-panel="1"] .plan-list,
[data-testid="column"]:has(.tex-agent-panel-marker) .plan-list {
    margin-bottom: 30px !important;
}

[data-tex-panel="1"] .ap-list,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-list {
    margin-bottom: 0 !important;
}

/* Plan buttons: aligned card style */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] {
    width: calc(100% - 28px) !important;
    max-width: calc(100% - 28px) !important;
    margin: 0 14px 25px 14px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button {
    position: relative !important;

    width: 100% !important;
    min-height: 72px !important;
    box-sizing: border-box !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 14px 18px 14px 58px !important;
    border-radius: 16px !important;

    background:
        radial-gradient(120% 90% at 100% 0%, rgba(255,255,255,0.075) 0%, transparent 58%),
        linear-gradient(145deg, rgba(255,255,255,0.080), rgba(255,255,255,0.045)) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;

    color: rgba(255,255,255,0.74) !important;
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.09) inset,
        0 12px 26px -22px rgba(0,0,0,0.58) !important;

    text-align: left !important;
    transition: all 0.16s ease !important;
}

[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button:hover,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button:hover {
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(255,255,255,0.13) 0%, transparent 58%),
        linear-gradient(145deg, rgba(255,255,255,0.115), rgba(255,255,255,0.060)) !important;
    border-color: rgba(255,255,255,0.24) !important;
    transform: translateY(-1px);
}

/* 선택된 플랜 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button[data-testid="baseButton-primary"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button[data-testid="baseButton-primary"] {
    background:
        radial-gradient(120% 90% at 100% 0%, rgba(179,25,34,0.20) 0%, transparent 56%),
        linear-gradient(145deg, rgba(255,255,255,0.155), rgba(255,255,255,0.075)) !important;
    border-color: rgba(255,255,255,0.30) !important;
    color: #FFFFFF !important;
    box-shadow:
        0 1px 0 0 rgba(255,255,255,0.12) inset,
        0 15px 30px -22px rgba(0,0,0,0.65) !important;
}

/* 왼쪽 고정 아이콘 공통 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button::before {
    content: "";
    position: absolute;
    left: 20px;
    top: 50%;
    transform: translateY(-50%);

    width: 26px;
    height: 26px;
    border-radius: 999px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    background: rgba(255,255,255,0.13);
    color: #FFFFFF;

    font-family: "bootstrap-icons" !important;
    font-size: 13px !important;
    font-weight: normal !important;
    line-height: 1 !important;
}

/* 플랜별 Bootstrap icon */
[data-tex-panel="1"] div[class*="st-key-plan_btn_basic"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_basic"] button::before {
    content: "\F46C"; /* lightning-charge-fill */
}

[data-tex-panel="1"] div[class*="st-key-plan_btn_plus"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_plus"] button::before {
    content: "\F3E6"; /* gem */
}

[data-tex-panel="1"] div[class*="st-key-plan_btn_premium"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_premium"] button::before {
    content: "\F589"; /* stars */
}

/* 버튼 내부 wrapper */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button div,
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button [data-testid="stMarkdownContainer"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button div,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: block !important;
    text-align: left !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* 실제 텍스트 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button p,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button p {
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    color: rgba(255,255,255,0.70) !important;
    font-family: "Sora", "Manrope", "Noto Sans KR", sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 650 !important;
    line-height: 1.48 !important;

    white-space: pre-line !important;
    text-align: left !important;
    letter-spacing: -0.01em !important;
}

/* 첫 줄: Basic / Plus / Premium */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button p::first-line,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button p::first-line {
    color: #FFFFFF !important;
    font-size: 14.5px !important;
    font-weight: 900 !important;
    line-height: 1.5 !important;
}

/* 선택된 카드 설명 줄 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] button[data-testid="baseButton-primary"] p,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] button[data-testid="baseButton-primary"] p {
    color: rgba(255,255,255,0.82) !important;
}

/* 플랜 카드 간격 조정 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] {
    margin: 0 14px 0px 14px !important;
}

/* Premium 아래와 다음 섹션 사이 간격 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_premium"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_premium"] {
    margin-bottom: 24px !important;
}

/* Final width normalization: 모든 박스 폭 통일 */
[data-tex-panel="1"] {
    --agent-box-x: 20px;
    --agent-box-w: calc(100% - 40px);
}

/* 플랜 버튼 outer wrapper는 추가 여백 제거 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] {
    width: 100% !important;
    max-width: 100% !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    padding-left: 0 !important;
    padding-right: 0 !important;
    box-sizing: border-box !important;
}

/* 모든 st.button 계열: 플랜 / 모델 / 초기화 / 재로드 */
[data-tex-panel="1"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton {
    width: var(--agent-box-w) !important;
    max-width: var(--agent-box-w) !important;
    margin-left: var(--agent-box-x) !important;
    margin-right: var(--agent-box-x) !important;
    box-sizing: border-box !important;
}

/* 사용 가능한 도구 / 예시 질문 / 상위 플랜 안내 */
[data-tex-panel="1"] .ap-list,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-list,
[data-tex-panel="1"] .ap-note-wrap,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-note-wrap {
    width: var(--agent-box-w) !important;
    max-width: var(--agent-box-w) !important;
    margin-left: var(--agent-box-x) !important;
    margin-right: var(--agent-box-x) !important;
    box-sizing: border-box !important;
}

/* 내부 박스는 부모 폭 100% */
[data-tex-panel="1"] .stButton > button,
[data-testid="column"]:has(.tex-agent-panel-marker) .stButton > button,
[data-tex-panel="1"] .ap-row,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-row,
[data-tex-panel="1"] .ap-note,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-note {
    width: 100% !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
}

/* 플랜 버튼 간격 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_"] .stButton {
    margin-bottom: 10px !important;
}

/* Premium 아래와 구분선 사이 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_premium"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_premium"] .stButton {
    margin-bottom: 24px !important;
}

/* Section title size up */
[data-tex-panel="1"] .ap-label,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-label {
    color: rgba(255,255,255,0.94) !important;
    font-family: "Sora", "Manrope", "Noto Sans KR", sans-serif !important;
    font-size: 17px !important;
    font-weight: 950 !important;
    letter-spacing: -0.01em !important;
    line-height: 1.25 !important;
    margin-left: 20px !important;
    margin-right: 20px !important;
}

/* Premium 아래 여백 줄이기 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_premium"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_premium"] {
    margin-bottom: 0 !important;
}

/* Premium 버튼 내부 stButton 여백도 제거 */
[data-tex-panel="1"] div[class*="st-key-plan_btn_premium"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-plan_btn_premium"] .stButton {
    margin-bottom: 6px !important;
}

/* 구독 플랜 → 챗봇 설정 사이 구분선만 간격 축소 */
[data-tex-panel="1"] .ap-hr.plan-to-settings,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-hr.plan-to-settings {
    margin: 12px 20px 8px 20px !important;
}

/* Model 표시 버튼은 disabled 상태여도 잘 보이게 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button {
    opacity: 1 !important;
    color: rgba(255,255,255,0.88) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.88) !important;
}

/* Model 버튼 내부 텍스트까지 같이 보정 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button *,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button * {
    opacity: 1 !important;
    color: rgba(255,255,255,0.88) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.88) !important;
}

/* Model 버튼 hover/클릭 느낌 제거 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button:hover,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button:hover {
    transform: none !important;
    cursor: default !important;
}

/* Premium일 때 안내 박스 대신 최소 간격만 유지 */
[data-tex-panel="1"] .ap-note-premium-gap,
[data-testid="column"]:has(.tex-agent-panel-marker) .ap-note-premium-gap {
    height: 40px !important;
}

/* 예시 질문 버튼: 기존 ap-row 카드처럼 보이게 */
[data-tex-panel="1"] div[class*="st-key-example_question_"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] {
    width: var(--agent-box-w) !important;
    max-width: var(--agent-box-w) !important;
    margin-left: var(--agent-box-x) !important;
    margin-right: var(--agent-box-x) !important;
    margin-bottom: 9px !important;
    box-sizing: border-box !important;
}

[data-tex-panel="1"] div[class*="st-key-example_question_"] button,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button {
    position: relative !important;

    width: 100% !important;
    min-height: 42px !important;
    box-sizing: border-box !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 10px 12px 10px 42px !important;
    border-radius: 13px !important;

    background: rgba(255,255,255,0.065) !important;
    border: 1px solid rgba(255,255,255,0.115) !important;

    color: rgba(255,255,255,0.86) !important;
    box-shadow: none !important;

    text-align: left !important;
    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 650 !important;
    line-height: 1.45 !important;
}

[data-tex-panel="1"] div[class*="st-key-example_question_"] button:hover,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button:hover {
    background: rgba(255,255,255,0.105) !important;
    border-color: rgba(255,255,255,0.18) !important;
    transform: translateY(-1px);
}

[data-tex-panel="1"] div[class*="st-key-example_question_"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button::before {
    content: "\F4BD"; /* chat-dots-fill */
    position: absolute;
    left: 14px;
    top: 50%;
    transform: translateY(-50%);

    width: 18px;
    height: 18px;
    border-radius: 999px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.82);

    font-family: "bootstrap-icons" !important;
    font-size: 10px !important;
    line-height: 1 !important;
}

[data-tex-panel="1"] div[class*="st-key-example_question_"] button p,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button p {
    margin: 0 !important;
    padding: 0 !important;
    color: inherit !important;
    font-family: inherit !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    line-height: inherit !important;
    text-align: left !important;
    white-space: normal !important;
    word-break: keep-all !important;
}

/* 예시 질문 버튼 최종 정렬 보정 */
[data-tex-panel="1"] div[class*="st-key-example_question_"]:not(.stButton),
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"]:not(.stButton) {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

/* 예시 질문 stButton wrapper 폭: ap-row와 동일하게 */
[data-tex-panel="1"] div[class*="st-key-example_question_"].stButton,
[data-tex-panel="1"] div[class*="st-key-example_question_"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"].stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] .stButton {
    width: var(--agent-box-w) !important;
    max-width: var(--agent-box-w) !important;
    margin-left: var(--agent-box-x) !important;
    margin-right: var(--agent-box-x) !important;
    margin-bottom: 9px !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}

/* 예시 질문 버튼 박스 */
[data-tex-panel="1"] div[class*="st-key-example_question_"] button,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button {
    position: relative !important;

    width: 100% !important;
    max-width: 100% !important;
    min-height: 42px !important;
    box-sizing: border-box !important;

    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;

    padding: 10px 12px 10px 42px !important;
    margin: 0 !important;

    border-radius: 13px !important;
    background: rgba(255,255,255,0.065) !important;
    border: 1px solid rgba(255,255,255,0.115) !important;
    box-shadow: none !important;

    color: rgba(255,255,255,0.86) !important;
    text-align: left !important;

    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 650 !important;
    line-height: 1.45 !important;
}

[data-tex-panel="1"] div[class*="st-key-example_question_"] button:hover,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button:hover {
    background: rgba(255,255,255,0.105) !important;
    border-color: rgba(255,255,255,0.18) !important;
    transform: translateY(-1px);
}

/* 왼쪽 채팅 아이콘 */
[data-tex-panel="1"] div[class*="st-key-example_question_"] button::before,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button::before {
    content: "\F4BD";
    position: absolute;
    left: 14px;
    top: 50%;
    transform: translateY(-50%);

    width: 18px;
    height: 18px;
    border-radius: 999px;

    display: inline-flex;
    align-items: center;
    justify-content: center;

    background: rgba(255,255,255,0.12);
    color: rgba(255,255,255,0.82);

    font-family: "bootstrap-icons" !important;
    font-size: 10px !important;
    line-height: 1 !important;
}

/* 버튼 내부 텍스트 wrapper 정렬 */
[data-tex-panel="1"] div[class*="st-key-example_question_"] button div,
[data-tex-panel="1"] div[class*="st-key-example_question_"] button [data-testid="stMarkdownContainer"],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button div,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button [data-testid="stMarkdownContainer"] {
    width: 100% !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    text-align: left !important;
}

/* 실제 질문 텍스트 */
[data-tex-panel="1"] div[class*="st-key-example_question_"] button p,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_"] button p {
    width: 100% !important;

    margin: 0 !important;
    padding: 0 !important;

    color: inherit !important;
    font-family: "Manrope", "Noto Sans KR", system-ui, sans-serif !important;
    font-size: 11.5px !important;
    font-weight: 650 !important;
    line-height: 1.45 !important;

    text-align: left !important;
    white-space: normal !important;
    word-break: keep-all !important;
    overflow-wrap: break-word !important;
}

/* 마지막 예시 질문 아래 여백 */
[data-tex-panel="1"] div[class*="st-key-example_question_5"].stButton,
[data-tex-panel="1"] div[class*="st-key-example_question_5"] .stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_5"].stButton,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-example_question_5"] .stButton {
    margin-bottom: 8px !important;
}

/* Model 표시 버튼 색상: 대화 초기화 / Agent 재로드와 동일하게 맞춤 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button {
    opacity: 1 !important;

    background: rgba(255,255,255,0.065) !important;
    border: 1px solid rgba(255,255,255,0.115) !important;

    color: rgba(255,255,255,0.86) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.86) !important;

    box-shadow: none !important;
    transform: none !important;
    cursor: default !important;
}

/* disabled 상태여도 텍스트 흐려지지 않게 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button *,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button * {
    opacity: 1 !important;
    color: rgba(255,255,255,0.86) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.86) !important;
}

/* hover해도 Model 박스 색상 변하지 않게 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button:hover,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button:hover {
    background: rgba(255,255,255,0.065) !important;
    border-color: rgba(255,255,255,0.115) !important;
    transform: none !important;
}

/* Model 버튼 disabled 배경 최종 보정 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button:disabled,
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button[disabled],
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button:disabled,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button[disabled] {
    opacity: 1 !important;

    background: rgba(255,255,255,0.065) !important;
    border: 1px solid rgba(255,255,255,0.115) !important;

    color: rgba(255,255,255,0.86) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.86) !important;

    box-shadow: none !important;
    transform: none !important;
    cursor: default !important;
}

/* Model 버튼 내부 텍스트도 disabled 흐림 제거 */
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button:disabled *,
[data-tex-panel="1"] div[class*="st-key-agent_model_display"] button[disabled] *,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button:disabled *,
[data-testid="column"]:has(.tex-agent-panel-marker) div[class*="st-key-agent_model_display"] button[disabled] * {
    opacity: 1 !important;
    color: rgba(255,255,255,0.86) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.86) !important;
}

</style>"""

_PANEL_JS = """<script>
(function () {
    function applyPanel() {
        try {
            var doc = window.parent.document;
            var m = doc.querySelector('.tex-agent-panel-marker');
            if (!m) return;
            var el = m;
            for (var i = 0; i < 16; i++) {
                el = el.parentElement;
                if (!el) return;
                var tid = el.getAttribute('data-testid');
                if (tid === 'column' || tid === 'stColumn') {
                    el.setAttribute('data-tex-panel', '1');
                    return;
                }
            }
        } catch (e) {}
    }
    applyPanel();
    [60, 200, 600, 1500].forEach(function(t){ setTimeout(applyPanel, t); });
    try {
        new MutationObserver(applyPanel).observe(
            window.parent.document.documentElement,
            { childList: true, subtree: true }
        );
    } catch(e) {}
})();
</script>"""




def render():
    """Streamlit 챗봇 페이지 렌더링. app.py에서 호출 가능."""
    try:
        st.set_page_config(page_title="TEX 2025 챗봇", page_icon="⚾", layout="wide")
    except st.errors.StreamlitAPIException:
        pass

    if not GEMINI_API_KEY and (not GCP_PROJECT_ID or not GCP_SERVICE_ACCOUNT_JSON):
        st.error(
            "인증 정보가 설정되어 있지 않습니다. "
            ".env에 GEMINI_API_KEY를 추가하거나, "
            "Streamlit Cloud Secrets에 GCP_PROJECT_ID와 GCP_SERVICE_ACCOUNT_JSON을 등록하세요."
        )
        st.stop()

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = None

    st.markdown(_PANEL_CSS, unsafe_allow_html=True)

    col_panel, col_chat = st.columns([1.18, 2.82], gap="medium")

    # ── 왼쪽 패널 ────────────────────────────────────────────
    with col_panel:
        st.markdown('<span class="tex-agent-panel-marker"></span>', unsafe_allow_html=True)
        # JS via iframe (components.html executes scripts; st.markdown does not)
        import streamlit.components.v1 as _components
        _components.html(_PANEL_JS, height=0, scrolling=False)

        # 구독 플랜
        st.markdown(
            '<div class="ap-label">구독 플랜</div><div class="ap-title-gap"></div>',
            unsafe_allow_html=True,
        )

        def _set_agent_tier(tier_key: str):
            st.session_state.current_tier = tier_key


        if "current_tier" not in st.session_state:
            st.session_state.current_tier = "basic"

        tier = st.session_state.current_tier

        for key in ["basic", "plus", "premium"]:
            config = TIER_CONFIG[key]

            label = (
                f'{config["label"]}\n'
                f'{config["tagline"]}'
            )

            st.button(
                label,
                key=f"plan_btn_{key}",
                use_container_width=True,
                type="primary" if tier == key else "secondary",
                on_click=_set_agent_tier,
                args=(key,),
            )

        tier = st.session_state.current_tier
        
        # 챗봇 설정
        st.markdown('<div class="ap-hr plan-to-settings"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ap-label">챗봇 설정</div><div class="ap-title-gap"></div>',
            unsafe_allow_html=True,
        )
        st.button(
            f" Model: {GEMINI_MODEL}",
            key="agent_model_display",
            use_container_width=True,
            disabled=True,
        )
        if st.button(" 대화 초기화", key="agent_reset_btn", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()
        if st.button(" Agent 재로드", key="agent_reload_btn", use_container_width=True):
            get_agent.clear()
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()

        # 사용 가능한 도구
        st.markdown('<div class="ap-hr"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ap-label">사용 가능한 도구</div><div class="ap-title-gap"></div>',
            unsafe_allow_html=True,
        )
        enabled = TIER_CONFIG[tier]["tools"]
        tool_rows = []
        for name, desc in TOOL_DESCRIPTIONS.items():
            is_enabled = name in enabled
            icon_html = (
                '<i class="bi bi-check-circle-fill"></i>'
                if is_enabled
                else '<i class="bi bi-lock-fill"></i>'
            )
            state_class = "" if is_enabled else " locked"

            tool_rows.append(
                f'<div class="ap-row{state_class}">'
                f'<span class="ap-icon">{icon_html}</span>'
                f'<div class="ap-body">'
                f'<span class="ap-name">{name}</span>'
                f'<span class="ap-desc">{desc}</span>'
                f'</div>'
                f'</div>'
            )
            
        st.markdown(
            '<div class="ap-list">' + "".join(tool_rows) + '</div>',
            unsafe_allow_html=True,
        )
        locked = [n for n in TOOL_DESCRIPTIONS if n not in enabled]

        if tier != "premium" and locked:
            st.markdown(
                f'<div class="ap-note-spacer-top"></div>'
                f'<div class="ap-note-wrap">'
                f'<div class="ap-note">'
                f'<i class="bi bi-unlock-fill"></i>'
                f'<span>상위 플랜에서 {len(locked)}개 도구 추가 활성화</span>'
                f'</div>'
                f'</div>'
                f'<div class="ap-note-spacer-bottom"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="ap-note-premium-gap"></div>',
                unsafe_allow_html=True,
            )

        # 예시 질문
        st.markdown('<div class="ap-hr"></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="ap-label">예시 질문</div><div class="ap-title-gap"></div>',
            unsafe_allow_html=True,
        )
        example_questions = [
            "균형점 시나리오 결과 알려줘",
            "TEX가 SEA 수준 선발진이었다면?",
            "9월 1점차 경기 승률은?",
            "역대 잔차 -9승 이하 팀 있어?",
            "K9 0.3σ 올리고 BB9 0.4σ 줄이면?",
            "SEA, LAD와 팀 성적 비교 레이더 차트 그려줘",
        ]
        
        def _set_example_question(question: str):
            st.session_state.pending_user_input = question


        for idx, question in enumerate(example_questions):
            st.button(
                question,
                key=f"example_question_{idx}",
                use_container_width=True,
                on_click=_set_example_question,
                args=(question,),
            )

    # ── 오른쪽 채팅 영역 ──────────────────────────────────────
    with col_chat:
        st.caption(
            f"현재 플랜: **{TIER_CONFIG[tier]['name']}** · "
            f"활성 도구 {len(enabled)}개 / {len(TOOL_DESCRIPTIONS)}개"
        )

        agent = get_agent(tier)

        def _queue_typed_input():
            text = st.session_state.get("agent_chat_input", "")
            if text:
                st.session_state.pending_user_input = text

        def _run_user_query(user_input: str):
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input,
            })

            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("분석 중..."):
                    try:
                        result = agent.run_sync(
                            user_input,
                            message_history=st.session_state.agent_history,
                        )
                    except Exception as e:
                        err_text = f"⚠️ Agent 호출 실패: {e}"
                        st.error(err_text)
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": err_text,
                            "tool_calls": [],
                        })
                        return

                st.markdown(result.output)

                tool_calls: list[dict] = []
                pending_calls: dict[str, dict] = {}

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

                st.session_state.agent_history = result.all_messages()
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": result.output,
                    "tool_calls": tool_calls,
                })

        # 1) 기존 대화 먼저 출력
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("tool_calls"):
                    _render_tool_calls(msg["tool_calls"], key_prefix=f"hist_{id(msg)}")

        # 2) 예시 질문 클릭 / 입력 제출로 들어온 질문 처리
        pending_input = st.session_state.pop("pending_user_input", None)

        if pending_input:
            _run_user_query(pending_input)

        # 3) 채팅 입력창은 항상 맨 마지막에 렌더링
        st.chat_input(
            "질문을 입력하세요...",
            key="agent_chat_input",
            on_submit=_queue_typed_input,
        )


if __name__ == "__main__":
    render()
