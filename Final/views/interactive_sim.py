"""Interactive Simulator — baseball_simulator.html 임베드.

Final/baseball_simulator.html (NSGA-II v5 12차원 σ Pareto 시뮬레이터)을
iframe으로 임베드. 슬라이더 + 캔버스 차트가 그대로 작동.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components


# HTML 위치: Final/baseball_simulator.html
# views/interactive_sim.py → ../../baseball_simulator.html (= Final/baseball_simulator.html)
FINAL_DIR = Path(__file__).resolve().parents[1]
_HTML_PATH = FINAL_DIR / "baseball_simulator.html"


def show() -> None:
    st.markdown("# 🎮 Interactive Simulator")
    st.markdown(
        """
        <div style="background:#F8FAFC;border-left:4px solid #003278;padding:14px 18px;margin:8px 0 18px;border-radius:6px;font-size:13.5px;color:#1B2435;line-height:1.7;">
            12차원 σ 슬라이더를 직접 조정하면 예상 시즌 승수가 즉시 갱신됩니다.<br>
            아래 시뮬레이터는 <b>NSGA-II v5 Pareto 50점</b>(baseline 89.8) 기반 선형 회귀 근사로,
            <b>aggressive·balanced·conservative</b> archetype 프리셋 버튼으로 빠르게 비교 가능합니다.<br>
            <span style="color:#94A3B8;">※ σ_norm > 0.10 영역은 학습 분포 외삽 경계 — 시뮬 가정 하에서만 유효 (시뮬레이터 안 주황색 음영).</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not _HTML_PATH.exists():
        st.error(f"파일을 찾을 수 없습니다: {_HTML_PATH}")
        st.caption("Final/baseball_simulator.html이 프로젝트 루트에 있는지 확인하세요.")
        return

    html_content = _HTML_PATH.read_text(encoding="utf-8")
    components.html(html_content, height=1700, scrolling=True)

    st.markdown("---")
    st.caption(
        "🛠 시뮬레이터 소스: `Final/baseball_simulator.html` (single-file HTML, 외부 의존성 없음). "
        "데이터: `nsga_pareto_phase8_v5_*.csv`에서 추출 + sklearn LinearRegression 회귀 계수."
    )
