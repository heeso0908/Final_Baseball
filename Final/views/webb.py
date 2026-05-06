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
**선발 투수 Webb - 삼진 vs 볼넷 해석**

### 한 줄 결론
Webb은 5명 중 투구폼 차이가 가장 뚜렷하게 보인 선수입니다.

### 쉽게 풀어보면
- 삼진을 잡을 때는 골반과 어깨가 잘 분리되고, 몸통 회전도 더 강하게 이어졌습니다.
- 반대로 볼넷 상황에서는 상체와 하체가 덜 분리되어, 힘이 순서대로 전달되는 흐름이 약해진 것으로 보입니다.
- 쉽게 말하면, 좋은 결과가 날 때는 몸이 "꼬였다가 풀리는" 순서가 더 안정적이었습니다.

### 그래서 무엇을 해야 하나?
- Webb은 폼 교정 또는 루틴 조정으로 개선 여지가 있는 케이스입니다.
- 특히 앞발이 닿는 순간의 상체-하체 분리, 릴리스 전 회전 타이밍을 안정화하는 코칭이 필요합니다.

### 의사결정 포인트
- 폼 교정: 우선순위 높음
- 운영 조정: 연속 등판 또는 부담 큰 상황에서 상태 확인 필요
- 권장 방향: 경기 전 루틴과 투구 메커니즘 체크포인트를 정해 일관성을 높이기
"""
    show_pitcher_page("Webb", "SO", "Walk", interpretation, key_findings)
