from shared import data
from views.pitcher import show_pitcher_page


def show():
    df = data['pitcher_ag']
    leiter_df = df[df['player'] == 'Leiter'].copy()
    top4 = leiter_df.reindex(leiter_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    interpretation = """
**선발 투수 Leiter - 삼진 vs 볼넷 해석**

### 한 줄 결론
Leiter의 볼넷 문제는 투구폼 하나로 설명하기 어렵습니다.

### 쉽게 풀어보면
- 삼진을 잡을 때와 볼넷을 내줄 때의 몸 움직임 차이가 뚜렷하게 크지는 않았습니다.
- 그래서 "폼이 무너져서 볼넷이 늘었다"고 단정하기는 어렵습니다.
- 오히려 구종 선택, 공이 들어가는 위치, 카운트 운영, 상대 타자와의 승부 방식이 더 큰 원인일 수 있습니다.

### 그래서 무엇을 해야 하나?
- 투구폼을 크게 고치기 전에, 불리한 카운트에서 어떤 공을 던졌는지 먼저 확인해야 합니다.
- 초구 스트라이크 비율, 볼넷 직전의 구종, 좌우 타자별 제구 패턴을 함께 봐야 합니다.

### 의사결정 포인트
- 폼 교정: 보조적으로 검토
- 피칭 디자인: 우선 검토
- 권장 방향: 카운트별 구종 조합과 제구 위치를 먼저 점검
"""
    show_pitcher_page("Leiter", "SO", "Walk", interpretation, key_findings)
