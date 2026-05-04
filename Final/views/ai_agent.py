import streamlit as st
from shared import page_hero


def show():
    page_hero(
        "AI Agent",
        "CSV 기반 분석 질의",
        "업로드된 CSV 파일을 기반으로 사용자가 궁금해하는 TEX 2025 잔차 분석 질문에 답합니다. 텍스트 요약뿐 아니라 그래프로 확인할 만한 구조화 결과도 함께 탐색하는 역할입니다.",
        [("Text Q&A", "white"), ("Chart-ready Data", "white"), ("Uploaded CSV", "white")],
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