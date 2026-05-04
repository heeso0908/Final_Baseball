from shared import data
from views.pitcher import show_pitcher_page


def show():
    df = data['pitcher_ag']
    webb_df = df[df['player'] == 'Webb'].copy()
    top4 = webb_df.reindex(webb_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    interpretation = """
**좌완 선발 Webb — 삼진 vs 볼넷 분기 메커니즘** ⭐

5명 중 **가장 명확한 차이**가 발견된 케이스.

### 핵심 차이

- **HSS at FP**: 삼진 23.0° vs 볼넷 11.4° (p=0.005, d=2.16)
  - 삼진 시 상체-하체 분리가 2배 이상 큼
- **Trunk/Hip ratio**: 삼진 2.51 vs 볼넷 1.49 (p=0.005, d=3.05)
  - 삼진 시 몸통이 골반보다 훨씬 빠르게 회전 (kinetic chain)
- **Trunk peak 3D**: 삼진 1377 vs 볼넷 1134°/s
  - 회전력 자체가 더 큼

### 메커니즘 해석


### 시사점

- **폼 미세조정 가능 영역**: HSS 안정화 코칭 가치 있음
- 좌완 선발의 특성상 **타석에 대한 분리 메커니즘**이 결정적
- 코칭/Pre-pitch 루틴으로 일관성 확보 가능
"""
    show_pitcher_page("Webb", "SO", "Walk", interpretation, key_findings)