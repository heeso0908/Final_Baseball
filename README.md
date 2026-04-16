# ⚾ Final_Baseball
내일배움캠프 데이터분석 최종프로젝트

MLB 야구 데이터를 활용해 분석 및 시각화를 수행하는 프로젝트입니다.  

---

## 👥 팀원 구성

- **이찬혁** : 리더
- **지소윤** : 부리더
- **김효준** : 서기
- **김희선** : 총무

---

## 📁 프로젝트 폴더 구조

```bash
Final_Baseball/
├── Data/                  # 원본 데이터 및 분석용 데이터 파일
├── Notebooks/             # EDA, 전처리, 모델링, 시각화용 Jupyter Notebook
├── Reports/               # 발표자료, 보고서, 결과 정리 문서
├── main.py                # 메인 실행 파일
├── pyproject.toml         # uv 프로젝트 설정 및 패키지 의존성 관리 파일
├── uv.lock                # 프로젝트 패키지 버전 고정 파일
├── .python-version        # 프로젝트 Python 버전 정보
└── README.md              # 프로젝트 소개 문서
```

---

## 🚀 환경 설정 (uv)

본 프로젝트는 [uv](https://docs.astral.sh/uv/)로 패키지를 관리합니다.

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

이후 VSCode 또는 Jupyter에서 커널을 `final-baseball (3.12.12)`로 선택하면 됩니다.

---

## 📁 파일 구조

```bash
Final_Baseball/
├── Data/                        # 원본 데이터
├── Notebooks/
│   └── 지소윤/
│       ├── 0.애매한 팀 찾기.ipynb
│       ├── 1.데이터수집.ipynb
│       ├── 2.데이터_전처리.ipynb    # 공통 전처리 파이프라인
│       ├── 3.EDA.ipynb
│       ├── 4.심층분석.ipynb
│       ├── 5.simulation.ipynb
│       ├── 6.ml.ipynb
│       ├── data/                  # 분석용 CSV (gitignore — 별도 공유)
│       └── baseball_kinematics/   # 투수 포즈 분석 서브 프로젝트
│           ├── pose_extractor_yolo.py   # YOLO + MediaPipe 포즈 추출
│           ├── pose_preprocessor.py     # 포즈 데이터 전처리
│           ├── release_detector.py      # 릴리즈 포인트 감지
│           ├── run_analysis.py          # 키네마틱 지표 계산
│           ├── run_batch_pose.py        # 배치 실행
│           ├── trim_videos.py           # 투구 영상 트리밍
│           ├── download_walks.py        # 볼넷 영상 다운로드
│           ├── download_strikeouts.py   # 삼진 영상 다운로드
│           ├── leiter_kinematics_analysis.ipynb  # 볼넷 vs 삼진 분석
│           └── _archive/               # 미사용 구버전 코드
├── Reports/
├── pyproject.toml               # 의존성 정의
├── uv.lock                      # 패키지 버전 고정
└── .python-version              # Python 3.12.12
```

> `data/`, `pose_output/`, `videos/` 폴더는 용량 문제로 git에서 제외됩니다. 팀 공유 드라이브를 통해 별도로 받아주세요.

---

## 📦 주요 패키지

- `pandas` / `numpy` / `scipy` — 데이터 처리
- `matplotlib` / `seaborn` / `plotly` — 시각화
- `scikit-learn` — 머신러닝
- `pybaseball` — MLB Statcast 데이터 수집
- `mediapipe` / `opencv-python` / `ultralytics` — 포즈 추출 (baseball_kinematics)
