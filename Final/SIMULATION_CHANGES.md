# 시뮬레이션 개선 내역

> 최종 업데이트: 2025년 5월  
> 기준 커밋: `217c93c` (Merge PR #11 from heeso0908/Final/agent)

---

## 1. 성능 최적화 (simulator.py)

### bisect 기반 샘플링 도입
- **기존**: `np.random.choice(events, p=weights)` — 매 타석마다 확률 리스트를 Python 레벨에서 재계산
- **변경**: `LineupPool` 생성 시 누적 가중치(cumulative weights)를 미리 계산, 타석 시뮬 시 `bisect.bisect()`로 O(log n) 탐색
- **효과**: 타격 PA 샘플링 3~5배 속도 향상 (100회 시뮬 기준 수 분 → 1분대 목표)
- `LineupPool.probs` 타입: `dict[str, dict]` → `dict[str, tuple[list, list]]` (events_list, cum_weights_list)
- `integrated_sim.py`의 `_simulate_inning_tracked`에도 동일하게 적용

### `_lineup_probs_as_dict` 헬퍼 추가
- bisect 최적화 후 `_build_player_projection_table`에서 probs를 dict처럼 접근해 발생하던 `TypeError: tuple indices must be integers` 버그 수정
- tuple → dict 복원 헬퍼 함수를 simulator.py에 추가

### 50시즌 상한 제거
- `_get_integrated_sim_result` 내 `min(n_sims, 50)` 제한 제거
- 사용자가 설정한 반복 횟수(100~1000회)가 그대로 적용됨
- UI 설명 문구도 수정: "최대 50시즌으로 제한" → 실제 설정값에 따름으로 변경

---

## 2. 기준선(Baseline) 불일치 수정 (shared.py, views/simulation.py)

### 문제
- 메인 시뮬레이션(100회): **87.2승** 표시
- 의사결정 보드 기준선(내부 10회): **82.5승** 표시
- 동일한 Baseline 2025 시나리오인데 다른 값이 표시되는 혼란

### 원인
- `get_live_scenario_results`가 내부적으로 `n_sims=10`으로 기준선을 별도 계산
- 노이즈가 많아 메인 시뮬과 다른 값 출력

### 해결
- `_render_decision_board`에 `baseline_override: float | None = None` 파라미터 추가
- Baseline 2025 시뮬 완료 시, 그 결과값(mean)을 의사결정 보드에 직접 전달
- 다른 시나리오의 delta(차이값)도 이 기준선 기반으로 재계산

---

## 3. 투수 연산 최적화 (shared.py)

### tex25 반복 로드 방지
- `_stats_to_pitcher_adj`가 호출마다 `_get_scenario_defaults()`를 재실행하는 문제
- Pareto 시나리오 6개 실행 시 6번 반복 → 불필요한 디스크 IO
- `tex25: dict | None = None` 파라미터 추가, 상위 함수에서 한 번 로드 후 전달

---

## 4. markov_pitching.py 최적화

- `_simulate_inning` 내 `list(state)` 불필요한 변환 2곳 제거
- `_adjust_for_leverage`, `_is_crisis` 모두 tuple 직접 수용 (내부에서 `sum(runners_on)` 사용)
- 변환 오버헤드 제거로 투수 시뮬 경량화

---

## 5. 선수 추적 기능 (shared.py, integrated_sim.py)

### `track_player_stats` 파라미터
- `_get_integrated_sim_result` 함수에 `track_player_stats: bool = True` 파라미터 추가
- `run_integrated_simulation` 호출 시 전달 → 선수별 이벤트 집계 활성화
- 반환 데이터: `player_projection`(타자), `pitcher_projection`(투수) DataFrame 포함

### 타자 성적 컬럼 (batter_df)
`선수`, `PA/시즌`, `H/시즌`, `2B/시즌`, `3B/시즌`, `HR/시즌`, `BB/시즌`, `K/시즌`, `AVG`, `OBP`, `SLG`, `OPS`

### 투수 성적 컬럼 (pitcher_df)
`투수`, `IP/시즌`, `BF/시즌`, `K/시즌`, `BB/시즌`, `HR/시즌`, `H/시즌`, `ER/시즌`, `ERA`, `WHIP`, `K%`, `BB%`

---

## 6. 시나리오별 선수 성적 비교 탭 교체 (views/simulation.py)

### 기존 방식의 문제
- `build_scenario_snapshots` (ERA 정적 스케일링) 사용
- 모든 시나리오가 동일한 `tex25["ERA"]`를 기준으로 계산 → 투수 수치가 시나리오 간 동일하게 표시

### 새 방식
- 실제 Markov 시뮬레이션 결과(`player_projection`, `pitcher_projection`) 직접 사용
- **현재 실행한 시나리오 vs Baseline 2025** 비교 구조
  - Baseline 2025 결과는 `@st.cache_data`로 캐시 → 이미 실행했으면 즉시 로드
- 타자: OPS 기준 막대 차트 (AVG/OBP/SLG/HR 툴팁)
- 투수: ERA 기준 막대 차트 (WHIP/K%/BB%/IP 툴팁)
- Baseline은 회색, 현재 시나리오는 레인저스 컬러로 시각 구분

---

## 7. 시나리오 정리 (simulator.py, views/simulation.py)

의사결정 보드 및 비교 차트에서 미사용 시나리오 제거:

| 상태 | 시나리오 |
|------|---------|
| 제거 | Hopeful Composite |
| 제거 | Langford Leap |
| 제거 | Risk Case |
| 제거 | Rotation Spike |
| 유지 | Baseline 2025 |
| 유지 | Bullpen Upgrade |
| 유지 | Hitter Boost |

---

## 8. 의사결정 보드 개선 (views/simulation.py)

- **조정 내역 컬럼 추가**: 후보 비교 테이블에 각 시나리오의 실제 수치 조정 내용 표시
- 기존 `get_scenario_snapshots` import 제거 (미사용)

---

## 9. 신규 파일

| 파일 | 역할 |
|------|------|
| `integrated_sim.py` | 타자(Markov) + 투수(markov_pitching) 통합 시뮬레이션 엔진 |
| `markov_pitching.py` | 투수 Markov 상태 전이 시뮬레이터 (선발/불펜 분리, 위기 상황 대응) |
| `nsga_search.py` | NSGA-II 기반 Pareto 최적 시나리오 탐색 (`run_integrated_simulation` 목적 함수) |

---

## 10. 알려진 구조

### Pareto 프론트 값 고정 관련
- `_PHASE8_CONFIGS`, `_PARETO_STAT_DELTAS`는 이전 NSGA-II 실행 결과로 하드코딩
- 시뮬레이션 로직 변경(속도 최적화 수준)은 결과에 영향 없음 → 재탐색 불필요
- 시뮬 로직 자체가 바뀌면 `nsga_search.py`로 재탐색 가능
