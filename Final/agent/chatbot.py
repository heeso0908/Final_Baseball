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

load_dotenv()

_PROMPT_PATH = Path(__file__).parent / "system_prompt.md"


def _load_system_prompt() -> str:
    if not _PROMPT_PATH.exists():
        return "당신은 TEX 2025 시즌 분석 보조 에이전트입니다. 한국어로 답변하세요."
    return _PROMPT_PATH.read_text(encoding="utf-8")


@st.cache_resource(show_spinner="ML 모델 학습 + Agent 초기화 중...")
def get_agent() -> Agent:
    """Agent 인스턴스를 1회만 생성해 세션 간 공유."""
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    model_id = f"google-gla:{model_name}"

    agent = Agent(
        model_id,
        system_prompt=_load_system_prompt(),
    )

    # 도구 등록 — docstring과 type hint가 자동으로 schema가 됨
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

    @agent.tool_plain
    def lookup_pareto(name: str) -> dict:
        """사전 계산된 Pareto/Grid 시나리오 결과 즉시 반환.

        유효 이름:
        - Pareto: 'aggressive', 'balanced', 'conservative' (또는 한국어 alias)
        - Grid: 'best_overall', 'best_bullpen', 'best_closegame',
                'best_pitching', 'worst_overall', 'baseline'
        """
        return tools.lookup_pareto(name)

    @agent.tool_plain
    def compare_team_2025(team: str) -> dict:
        """TEX 2025와 다른 팀 2025 투수 통계 비교.

        팀 코드(SEA/HOU/ATH/LAA 등) 또는 한국어 별명(매리너스, 애슬래틱스 등) 지원.
        """
        return tools.compare_team_2025(team)

    @agent.tool_plain
    def swap_team_pitching(team: str) -> dict:
        """대상 팀 2025 투수 통계를 TEX 입력에 이식해 시뮬.

        "TEX가 SEA 수준 선발진이었다면 몇 승?" 형태 질문에 답한다.
        K/9, BB/9, HR/9, BABIP를 σ로 변환해 estimate_residual_scenario 호출.
        """
        return tools.swap_team_pitching(team)

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
    st.caption(
        "반사실 시뮬레이션 · 경쟁팀 비교 · 게임로그 조회 · historical 비교 — "
        "6개 도구로 TEX 2025 시즌을 다각도 분석합니다."
    )

    # API 키 확인
    if not os.getenv("GEMINI_API_KEY"):
        st.error(
            "환경 변수 `GEMINI_API_KEY`가 설정되어 있지 않습니다. "
            "프로젝트 루트의 `.env` 파일을 확인하세요."
        )
        st.stop()

    agent = get_agent()

    # 세션 상태 — 대화 메시지 + Agent 메시지 히스토리
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []  # UI 표시용 [{role, content}]
    if "agent_history" not in st.session_state:
        st.session_state.agent_history = None  # PydanticAI 메시지 히스토리

    # 사이드바 — 컨트롤
    with st.sidebar:
        st.markdown("### 챗봇 설정")
        st.caption(f"Model: `{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}`")
        if st.button("🗑️ 대화 초기화", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()

        if st.button("🔄 Agent 재로드 (.env/프롬프트 변경 후)", use_container_width=True):
            get_agent.clear()
            st.session_state.chat_messages = []
            st.session_state.agent_history = None
            st.rerun()

        st.markdown("---")
        st.markdown("### 사용 가능한 도구")
        st.caption(
            "- `estimate_residual_scenario` — σ 점추정\n"
            "- `lookup_pareto` — 사전 시나리오\n"
            "- `compare_team_2025` — 팀 비교\n"
            "- `swap_team_pitching` — 이식 시뮬\n"
            "- `query_gamelog` — 게임로그\n"
            "- `query_team_history` — historical"
        )

        st.markdown("---")
        st.markdown("### 예시 질문")
        st.caption(
            "- 균형점 시나리오 결과 알려줘\n"
            "- TEX가 SEA 수준 선발진이었다면?\n"
            "- 9월 1점차 경기 승률은?\n"
            "- 역대 잔차 -9승 이하 팀 있어?\n"
            "- K9 0.3σ 올리고 BB9 0.4σ 줄이면?"
        )

    # 기존 대화 표시
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("tool_calls"):
                with st.expander(f"🔧 도구 호출 내역 ({len(msg['tool_calls'])}건)"):
                    for tc in msg["tool_calls"]:
                        st.markdown(f"**{tc['name']}**")
                        st.code(f"args: {tc['args']}", language="python")
                        if tc.get('result') is not None:
                            st.code(f"result: {tc['result']}", language="python")

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
                err_text = f"❌ Agent 호출 실패: {e}"
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
            with st.expander(f"🔧 도구 호출 내역 ({len(tool_calls)}건)"):
                for tc in tool_calls:
                    st.markdown(f"**{tc['name']}**")
                    st.code(f"args: {tc['args']}", language="python")
                    if tc.get('result') is not None:
                        st.code(f"result: {tc['result']}", language="python")

        # 세션 상태 업데이트
        st.session_state.agent_history = result.all_messages()
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": result.output,
            "tool_calls": tool_calls,
        })


if __name__ == "__main__":
    render()
