# ⚾ Final_Baseball

내일배움캠프 데이터분석 최종프로젝트  
**2025 텍사스 레인저스의 피타고리안 기대 승률과 실제 승률 괴리 원인 분석**

MLB 야구 데이터를 활용해 2025년 텍사스 레인저스의 시즌 성과를 분석하고,  
피타고리안 기대 승수 대비 실제 승수가 낮게 나타난 원인을 통계, 머신러닝, 시뮬레이션, 시각화 관점에서 해석하는 프로젝트

---

## 👥 팀원 구성

| 이름 | 역할 |
|---|---|
| 이찬혁 | 리더 |
| 지소윤 | 부리더 |
| 김효준 | 서기 |
| 김희선 | 총무 |

---

## 🎯 프로젝트 주제

2025년 텍사스 레인저스는 실제 성적 기준 **81승 81패**를 기록했으나,  
득점과 실점을 기반으로 계산한 피타고리안 기대 승수는 약 **90승** 수준

즉, 득실차 기준으로는 더 많은 승리를 거둘 수 있었던 팀이었지만,  
실제 결과는 기대치보다 낮게 나타남

본 프로젝트의 핵심 질문은 다음과 같음

> **“2025년 텍사스 레인저스는 왜 피타고리안 기대 승수보다 실제 승수가 낮았을까?”**

이를 설명하기 위해 팀 레벨 지표, 접전 경기 성과, 불펜 운영, 타선 생산성, 수비 지표,  
머신러닝 기반 잔차 분석, 시나리오 시뮬레이션을 종합적으로 활용

---

## 📌 분석 목표

- 2025년 텍사스 레인저스의 실제 승수와 피타고리안 기대 승수 차이 정의
- 기대 승수 대비 실제 승수 괴리, 즉 잔차를 설명하는 주요 요인 탐색
- 전체 MLB 팀 데이터를 활용한 머신러닝 기반 잔차 모델 구축
- 접전 경기, 불펜, 수비, 득점 기복 등 운영 지표 중심의 시나리오 구성
- Markov Chain과 Monte Carlo Simulation을 활용한 시즌 재구성
- Grid Search, NSGA-II, Bayesian Optimization 기반 시나리오 탐색
- Tableau와 Streamlit을 활용한 분석 결과 시각화

---

## 🧠 핵심 분석 방향

본 프로젝트는 단순 승패 결과 요약이 아니라,  
**“기대 승수 대비 실제 승수의 괴리”**를 중심으로 2025 시즌을 재해석

분석 흐름은 다음과 같음

1. **피타고리안 기대 승수 계산**
   - 득점과 실점을 기반으로 기대 승률 및 기대 승수 산출

2. **잔차 정의**
   - `Residual = Actual Wins - Pythagorean Wins`

3. **팀 레벨 EDA**
   - 월별 성적
   - 홈/원정 성적
   - 1점 차 경기
   - 접전 경기
   - 불펜 및 세이브 상황
   - 타선 생산성
   - 수비 지표

4. **머신러닝 기반 잔차 분석**
   - Ridge, Lasso, Random Forest, XGBoost 등 활용
   - SHAP 및 Feature Importance 기반 주요 변수 해석
   - 피타고리안 기대 승수에 이미 반영되는 득점/실점 요약 변수와 운영 지표 분리

5. **시나리오 기반 시즌 재구성**
   - 불펜 안정성 개선
   - 접전 경기 성과 개선
   - 수비 및 운영 지표 변화
   - 잔차 변화 기반 예상 승수 비교

6. **최적화 기반 시나리오 탐색**
   - Signed Proxy 기반 Grid Search
   - Pareto Front 기반 다목적 최적화
   - Optuna 기반 Bayesian Optimization

---

## 📁 프로젝트 폴더 구조
```bash
Final_Baseball/
├── Data/                        # 원본 데이터 및 분석용 CSV 파일
├── Final/                       # 최종 제출 노트북 및 결과물
│   ├── 1.데이터수집.ipynb
│   ├── 2.데이터_전처리.ipynb       # 공통 전처리 파이프라인
│   ├── 3.EDA.ipynb               # 탐색적 데이터 분석
│   ├── 4.심층분석.ipynb            # 접전 경기 · HR 의존도 등 심층 분석
│   ├── 5.통합본(0428+ML 수정)_v3.ipynb  # Markov + MC 시뮬레이션 + ML 잔차 보정 통합본
│   │                             #   ├ 1~4. 데이터 로드 · 베이스 엔진 · 타자 확률 · 타순 구성
│   │                             #   ├ 5~8. 시뮬레이션 함수 · 기본 시뮬 · ML 잔차 모델 · 통합 파이프라인
│   │                             #   ├ 9~11. 시나리오 정의 · 실행 · 스케줄 구성 요소
│   │                             #   ├ 12. Signed Proxy 그리드 탐색 & 자동 시나리오 선정
│   │                             #   ├ 13. Pareto Front — 다목적 최적화 (NSGA-II)
│   │                             #   └ 14. Bayesian Optimization (Optuna) — Grid 대체 실험
│   └── output/                   # 시뮬레이션 결과물 (gitignore)
│       ├── signed_proxy_scenario_summary.csv  # 시나리오별 승수 요약
│       ├── pareto_summary.csv                 # Pareto front 대표 시나리오
│       ├── pareto_front.png
│       └── bayesian_vs_grid.png
├── Notebooks/                   # 팀원별 작업 노트북
├── Reports/                     # 팀원별 작업 로그
├── pyproject.toml               # 의존성 정의
├── uv.lock                      # 패키지 버전 고정
└── .python-version              # Python 3.12.12
```

> `Data/` 폴더 CSV는 git에 포함  
> `Final/output/`의 시뮬레이션 결과 CSV는 gitignore 처리

---

## 📦 주요 패키지

- `pandas` / `numpy` / `scipy` — 데이터 처리
- `matplotlib` / `seaborn` / `plotly` — 시각화
- `scikit-learn` / `xgboost` — 머신러닝 · 잔차 모델
- `pybaseball` — MLB Statcast 데이터 수집
- `pymoo` — 다목적 최적화 (NSGA-II)
- `optuna` — Bayesian Optimization

---

## 🚀 환경 설정 (uv)

[uv](https://docs.astral.sh/uv/)로 패키지 관리

### uv 설치 (처음 한 번만)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 프로젝트 환경 세팅
```bash
# 저장소 클론 후
git clone <repo-url>
cd Final_Baseball

# 가상환경 생성 + 패키지 설치 (한 번에)
uv sync
```

### Jupyter 커널 등록
```bash
.venv/bin/python -m ipykernel install --user --name final-baseball --display-name "final-baseball (3.12.12)"
```

이후 VS Code 또는 Jupyter에서 커널을 `final-baseball (3.12.12)`로 선택

---