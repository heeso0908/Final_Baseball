import streamlit as st
from shared import page_hero


def show():
    page_hero(
        "AI Agent",
        "잔차·시나리오 질의 보조",
        "TEX 2025 잔차 원인, 하이 레버리지 부진 케이스 모션 근거, 수동/Grid/Pareto 시나리오 후보를 같은 맥락에서 조회합니다. 화면에서 본 의사결정 후보를 다시 확인하거나 특정 조합의 예상 승수와 delta를 물어볼 수 있습니다.",
        [("Scenario Lookup", "white"), ("Optimization Summary", "white"), ("Team Compare", "white")],
    )
    try:
        from agent import chatbot
    except ModuleNotFoundError as exc:
        st.error(f"AI Agent 의존성을 불러오지 못했습니다: {exc}")
        st.info("필요 패키지: `pydantic-ai`, `python-dotenv`")
        return
    except Exception as exc:
        st.error(f"AI Agent 초기화 중 오류가 발생했습니다: {type(exc).__name__}: {exc}")
        return

    chatbot.render()
