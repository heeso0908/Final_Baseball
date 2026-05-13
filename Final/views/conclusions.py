import streamlit as st
from shared import (
    page_hero, finding_box,
    REPORT_FINDINGS, build_player_report_pdf, build_team_report_pdf,
)

_HR = "<hr style='margin: 44px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>"


def show():
    page_hero(
        "Conclusions",
        "핵심 결론 및 의사결정 가이드",
        "시뮬레이션, 선수 비교, 개별 동작 분석 순으로 결론을 정리합니다.<br>"
        "특정 투수 한 명의 폼이 원인이 아니라, 접전 경기 운영·불펜 활용·선수 컨디션이 함께 만들어 낸 -9.1승 잔차입니다.",
        [("Residual Summary", "white"), ("Decision Candidates", "white"), ("PDF", "white")],
    )

    # ── 1. 시뮬레이션 결론 ───────────────────────────────────────────────────
    st.markdown("## 1. 시뮬레이션 결론")

    finding_box(
        "-9.1승 잔차 중 약 70%는 경기 운영 방식으로 설명 가능",
        "162경기 × 20시즌 시뮬레이션 결과, 평균 83.95승이 나왔습니다.<br>"
        "기대치 90.1승에서 시뮬레이션 평균 83.95승까지의 차이인 약 6.2승(약 68%)은 접전 상황에서의 투수 약화·마무리 투입 타이밍·불펜 공백 등으로 설명됩니다.<br>"
        "실제 81승까지 남는 약 3.0승은 운이나 아직 발견하지 못한 요인으로 보입니다."
    )

    mech_col, driver_col = st.columns([1, 1], gap="large")
    with mech_col:
        st.markdown("""
        <div class="glass-card glass-card-navy">
            <div class="chart-title">승수 차이를 만든 원인</div>
            <div class="chart-caption">
                <b>접전 상황 마무리 투수 삼진율 저하</b><br>
                1점차 경기에서 삼진 비율이 약 11.6%p 낮아지는 패턴<br><br>
                <b>위기 상황에서 마무리 투수 조기 투입</b><br>
                이닝 중간에 투입 시 상대 타자와의 궁합이 맞지 않는 경우 발생<br><br>
                <b>마무리 투수 공백 기간</b><br>
                Jackson 방출(7/23) → Maton 합류(8/1) 사이 공백 반영<br><br>
                <b>Garcia의 접전 경기 피안타율 0.625</b><br>
                1점차 마무리 기용 시 특정 타자 유형에 약점
            </div>
        </div>
        """, unsafe_allow_html=True)

    with driver_col:
        st.markdown("""
        <div class="glass-card glass-card-red">
            <div class="chart-title">승수에 가장 큰 영향을 준 요인 순위</div>
            <div class="chart-caption">
                <b>타자 단타 증가</b> &nbsp; 상관관계 +0.91 ★★★<br>
                <b>타자 삼진 감소</b> &nbsp;&nbsp;&nbsp; 상관관계 −0.84 ★★★<br>
                <b>선발 피홈런 감소</b> 상관관계 −0.80 ★★★<br>
                <b>팀 전체 삼진율</b> &nbsp; 상관관계 +0.59 ★★<br><br>
                머신러닝 분석과 일치하는 핵심 요인: <b>선발 피홈런 감소</b><br>
                시뮬레이션에서 새로 발견: <b>타자 단타 증가 · 삼진 감소</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card glass-card-accent">
        <div class="chart-title">머신러닝 분석과 시뮬레이션 결과 비교</div>
        <div class="chart-caption">
            <b>두 분석 모두 일치:</b> 선발 피홈런 감소 — 가장 강력하고 안정적인 투수 개선 요인<br>
            <b>방향은 같지만 크기 다름:</b> 볼넷 감소 / 마무리 볼넷 비율<br>
            <b>시뮬레이션에서만 발견:</b> 타자 단타·삼진은 머신러닝이 포착하지 못한 영역<br>
            <b>머신러닝이 이미 반영:</b> 세이브 성공률 / 1점차 승률 / 이닝 중 승계 주자 처리율<br><br>
            두 분석은 서로 모순이 아니라 <b>보완 관계</b>입니다. 머신러닝은 원인 분석, 시뮬레이션은 개선 경로를 담당합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quote-card">
        <p><strong>시뮬레이션 핵심 메시지</strong></p>
        <p>
        -9.1승 잔차 중 약 6.2승은 경기 운영 방식에서 설명되는 <b>발견된 원인</b>입니다.<br>
        나머지 약 3.0승과 그 이상의 향상도 현실적인 변화 범위 내에서 만회 가능한 조합이 존재합니다.<br>
        가장 효과적인 개선 방향은 <b>타자 단타 증가</b>와 <b>선발 피홈런 감소</b>입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── 2. Comparison — 선수 모션 분석 요약 ─────────────────────────────────
    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 2. 선수 투구 동작 분석 비교 요약")

    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">분석 관점</div>
        <div class="chart-caption">
            하이 레버리지 상황에서 부진했던 투수 4명과 비교 기준 Armstrong, <b>총 5명의 투구 동작을 비교</b>했습니다.<br>
            잘 던질 때와 못 던질 때의 골반·어깨·몸통 움직임 차이를 3D로 측정해,
            코칭으로 개선할 수 있는 선수와 기용 방식을 바꿔야 하는 선수를 나눕니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _PITCHER_CARDS = [
        {
            "name": "Armstrong",
            "type": "비교 기준",
            "type_color": "#166534",
            "bg": "rgba(240,253,244,0.97)",
            "border": "rgba(22,101,52,0.22)",
            "finding": "불펜에서 가장 안정적이었던 투수",
            "action": "다른 선수들의 기준점으로 사용",
        },
        {
            "name": "Webb",
            "type": "동작 차이 뚜렷",
            "type_color": "#003278",
            "bg": "rgba(240,245,255,0.97)",
            "border": "rgba(0,50,120,0.22)",
            "finding": "잘 던질 때와 못 던질 때<br>골반·어깨 분리와 몸통 회전이<br>눈에 띄게 달라짐",
            "action": "코칭이나 경기 전 루틴 조정으로<br>개선 가능성 있음",
        },
        {
            "name": "Garcia",
            "type": "동작 차이 없음",
            "type_color": "#9A3412",
            "bg": "rgba(255,247,237,0.97)",
            "border": "rgba(154,52,18,0.22)",
            "finding": "좋을 때와 나쁠 때의 투구 동작이 거의 같음",
            "action": "폼보다 투입 타이밍,<br>상대 타자 매치업을 먼저 검토",
        },
        {
            "name": "Leiter",
            "type": "동작 차이 없음",
            "type_color": "#9A3412",
            "bg": "rgba(255,247,237,0.97)",
            "border": "rgba(154,52,18,0.22)",
            "finding": "볼넷이 잦지만<br>투구 동작 자체의 차이는 크지 않음",
            "action": "구종 구성과 카운트별 제구 위치를 중점 점검",
        },
        {
            "name": "Jackson",
            "type": "투구 패턴 재해석",
            "type_color": "#4A5568",
            "bg": "rgba(247,250,252,0.97)",
            "border": "rgba(74,85,104,0.22)",
            "finding": "사이드암이 아닌<br>몸을 기울인 오버핸드 패턴으로 재해석됨",
            "action": "폼 교정보다 유리한 상황(타자 유형)에<br>맞게 쓰는 방향이 현실적",
        },
    ]

    comp_cols = st.columns(5, gap="small")
    for col, card in zip(comp_cols, _PITCHER_CARDS):
        with col:
            st.markdown(f"""
            <div style="background:{card['bg']};border:1px solid {card['border']};border-radius:14px;
                        padding:18px 12px 16px;text-align:center;min-height:214px;">
                <div style="display:inline-block;background:{card['type_color']};color:#fff;
                            font-size:11.5px;font-weight:800;padding:3px 11px;border-radius:20px;
                            margin-bottom:10px;letter-spacing:0.4px;white-space:nowrap;">
                    {card['type']}
                </div>
                <div style="font-size:20px;font-weight:800;color:#0D1B33;margin-bottom:6px;">
                    {card['name']}
                </div>
                <div style="font-size:13.5px;font-weight:800;color:{card['type_color']};margin-bottom:8px;line-height:1.45;">
                    {card['finding']}
                </div>
                <div style="font-size:13px;color:#475569;line-height:1.55;">
                    → {card['action']}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── 3. 선수별 운영 권고 ──────────────────────────────────────────────────
    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 3. 선수별 운영 권고")

    st.markdown("""
    <div class="glass-card" style="margin-bottom:24px;">
        <div class="chart-title">코칭 vs 기용 방식 조정</div>
        <div class="chart-caption">
            모든 선수에게 같은 처방을 적용하지 않습니다.<br>동작 근거가 있는 선수는 코칭, 그렇지 않은 선수는 기용 방식을 먼저 검토합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    _REC_ROWS = [
        ("Webb",      "동작 차이 뚜렷",   "rgba(13,27,51,0.08)",    "동작 차이가 가장 뚜렷함. 코칭 개입 가능성 높음",               "상체-하체 분리와 릴리스 타이밍 안정화",                "높음"),
        ("Leiter",    "동작 차이 없음",   "rgba(179,25,34,0.08)",   "동작 차이는 제한적. 구종 구성과 제구 위치 점검 필요",           "카운트별 구종 조합과 제구 위치 점검",                  "중간"),
        ("Garcia",    "동작 차이 없음",   "rgba(179,25,34,0.08)",   "동작 차이는 작음. 마무리 기용 시점과 상대 타자 재검토 필요",   "상대 타자 유형, 연투 여부, 좌우 타자 기준 등판 조정", "높음"),
        ("Armstrong", "비교 기준",        "rgba(47,158,101,0.10)",  "확정적 결론은 어려움. 등판 간격과 피로도 관리 확인 필요",      "등판 전 준비 시간과 연투 후 성적 관리",                "중간"),
        ("Jackson",   "투구 패턴 재해석", "rgba(184,189,199,0.15)", "폼 교정보다 상대 타자별 유불리 상황에 맞게 쓰는 것이 효과적", "좌우 타자별 강점 파악 후 유리한 상황에 집중 활용",    "중간"),
    ]
    _PRIORITY_COLOR = {"높음": "#B31922", "중간": "#243A5E"}

    _td = "padding:11px 14px;"
    rows_html = "".join(
        f'<tr style="background:{bg};border-bottom:1px solid #E4E8EF;">'
        f'<td style="{_td}font-weight:700;color:#0D1B33;white-space:nowrap;">{name}</td>'
        f'<td style="{_td}color:#334155;white-space:nowrap;">{ptype}</td>'
        f'<td style="{_td}color:#334155;">{interp}</td>'
        f'<td style="{_td}color:#334155;">{rec}</td>'
        f'<td style="{_td}font-weight:700;color:{_PRIORITY_COLOR.get(priority,"#667085")};text-align:center;white-space:nowrap;">{priority}</td>'
        f'</tr>'
        for name, ptype, bg, interp, rec, priority in _REC_ROWS
    )
    import streamlit.components.v1 as components
    _th = "padding:11px 14px;font-weight:800;color:#1B2435;"
    html_content = (
        '<style>*{margin:0;padding:0;box-sizing:border-box;font-family:Manrope,Pretendard,sans-serif;}</style>'
        '<div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:10px;align-items:center;">'
        '<span style="font-size:14px;color:#1B2435;font-weight:700;">행 색상 기준</span>'
        '<span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#1B2435;"><span style="width:12px;height:12px;border-radius:3px;background:rgba(47,158,101,0.25);display:inline-block;"></span>비교 기준</span>'
        '<span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#1B2435;"><span style="width:12px;height:12px;border-radius:3px;background:rgba(13,27,51,0.18);display:inline-block;"></span>동작 차이 뚜렷</span>'
        '<span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#1B2435;"><span style="width:12px;height:12px;border-radius:3px;background:rgba(179,25,34,0.18);display:inline-block;"></span>동작 차이 없음</span>'
        '<span style="display:flex;align-items:center;gap:6px;font-size:12.5px;color:#1B2435;"><span style="width:12px;height:12px;border-radius:3px;background:rgba(184,189,199,0.35);display:inline-block;"></span>투구 패턴 재해석</span>'
        '</div>'
        f'<div style="border-radius:12px;overflow:hidden;border:1px solid #E4E8EF;box-shadow:0 4px 16px -8px rgba(13,27,51,0.12);">'
        f'<table style="width:100%;border-collapse:collapse;font-size:14px;">'
        f'<thead><tr style="background:#F3F5F8;border-bottom:2px solid #E4E8EF;">'
        f'<th style="{_th}text-align:left;">선수</th>'
        f'<th style="{_th}text-align:left;">유형</th>'
        f'<th style="{_th}text-align:left;">핵심 해석</th>'
        f'<th style="{_th}text-align:left;">권고</th>'
        f'<th style="{_th}text-align:center;">우선순위</th>'
        f'</tr></thead>'
        f'<tbody>{rows_html}</tbody>'
        f'</table></div>'
    )
    components.html(html_content, height=320, scrolling=False)

    # ── 4. 종합 결론 ─────────────────────────────────────────────────────────
    st.markdown("<hr style='margin: 16px 0 44px 0; border:none; border-top:1px solid #E2E8F0;'>", unsafe_allow_html=True)
    st.markdown("## 4. 종합 결론")

    st.markdown("""
    <div class="glass-card glass-card-accent">
        <div class="chart-title">구단 의사결정과의 연결</div>
        <div class="chart-caption">
            이 분석은 "누구의 폼을 고칠 것인가"만 답하는 자료가 아닙니다.<br>
            어떤 선수는 코칭이 필요하고, 어떤 선수는 투입 상황을 바꿔야 하며, 어떤 경우에는 외부 보강을 검토해야 합니다.<br><br>
            최종 판단은 선수별 동작 분석 결과와 Simulation 페이지의 개선안 순위표를 함께 보고 내려야 합니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="quote-card">
        <p><strong>종합 결론</strong></p>
        <p>
        TEX 2025의 -9.1승 잔차는 한 가지 이유로 설명되지 않습니다.<br>
        Webb처럼 투구 폼 조정으로 개선 가능성이 보이는 선수도 있지만, Garcia나 Jackson처럼 기용 방식과 상대 타자 매치업을 먼저 봐야 하는 선수도 있습니다.<br>
        최종 권고는 <b>불펜 운영 개선, 선수별 코칭, 상황에 맞는 투수 기용, 필요 시 외부 보강</b>을 함께 검토하는 것입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
    with st.expander("**분석 한계와 다음 단계**", expanded=False):
        st.markdown("""
        <div style="font-size:14px; line-height:1.85; color:#334155; padding:10px 6px;">

        <b>데이터 한계</b><br>
        - 선수별 영상 표본이 많지 않아 모든 차이를 확정적으로 말하기는 어렵습니다.<br>
        - 방송 영상 기반이므로 카메라 각도와 프레임 수의 한계가 있습니다.<br><br>

        <b>모델 한계</b><br>
        - 사용한 AI 동작 분석 모델(MotionAGFormer)은 야구 전용이 아니므로, 수치는 의사결정의 보조 근거로 활용하는 것이 적절합니다.<br><br>

        <b>다음 단계</b><br>
        - 구종(직구·변화구 등)별로 나눠 다시 비교하기<br>
        - 시즌 전체 데이터를 길게 추적하기<br>
        - 좌타자/우타자 상대 성적과 등판 상황을 함께 연결하기

        </div>
        """, unsafe_allow_html=True)

    # ── 5. 보고서 출력 ───────────────────────────────────────────────────────
    st.markdown(_HR, unsafe_allow_html=True)
    st.markdown("## 5. 보고서 출력")

    st.markdown("""
    <div class="glass-card">
        <div class="chart-title">PDF 보고서</div>
        <div class="chart-caption">
            선수별 보고서와 팀 전체 보고서를 PDF로 출력합니다.<br>
            승수 차이 원인, 선수별 분석 근거, 개선 시나리오가 같은 흐름으로 정리됩니다.
        </div>
    </div>
    """, unsafe_allow_html=True)

    report_col_1, report_col_2 = st.columns([1, 1], gap="large")
    with report_col_1:
        st.markdown("""
        <div class="glass-card glass-card-amber">
            <div class="chart-title">선수별 요약 보고서</div>
            <div class="chart-caption">
                선택한 선수의 핵심 지표, 쉬운 해석, 코칭 또는 기용 판단 포인트를 정리합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <style>
        div[class*="st-key-player_report_select"] label p {
            font-size: 17px !important;
            font-weight: 700 !important;
            color: #1B2435 !important;
        }
        </style>
        """, unsafe_allow_html=True)
        selected_report_player = st.selectbox(
            "선수별 보고서 대상",
            list(REPORT_FINDINGS.keys()),
            key="player_report_select",
        )
        try:
            _player_pdf = build_player_report_pdf(selected_report_player)
            st.download_button(
                "선수별 보고서 PDF 출력",
                data=_player_pdf,
                file_name=f"tex_2025_player_report_{selected_report_player.lower()}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _e:
            st.error(f"PDF 생성 오류: {_e}")

    with report_col_2:
        st.markdown("""
        <div class="glass-card glass-card-amber">
            <div class="chart-title">팀 요약 보고서</div>
            <div class="chart-caption">
                팀 승수 차이 요약, 팀 장단점, 선수별 요약, 최종 결론을 정리합니다.
            </div>
        </div>
        """, unsafe_allow_html=True)
        try:
            _team_pdf = build_team_report_pdf()
            st.download_button(
                "팀 요약 보고서 PDF 출력",
                data=_team_pdf,
                file_name="tex_2025_team_residual_summary.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        except Exception as _e:
            st.error(f"PDF 생성 오류: {_e}")
