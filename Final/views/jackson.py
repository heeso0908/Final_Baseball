from shared import data
from views.pitcher import show_pitcher_page


def show():
    df = data['pitcher_ag']
    jack_df = df[df['player'] == 'Jackson'].copy()
    top4 = jack_df.reindex(jack_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    interpretation = """
**불펜 투수 Jackson - 세이브 vs 블론 세이브 해석**

### 한 줄 결론
Jackson의 투구폼은 "문제 있는 사이드암"이라기보다, 몸을 옆으로 조금 기울여 던지는 오버핸드에 가깝습니다.

### 쉽게 풀어보면
- 일부 수치만 보면 팔 각도가 낮아 보여서 사이드암처럼 보일 수 있습니다.
- 하지만 전체 움직임을 보면, 팔을 옆으로 빼서 던지는 투수라기보다는 상체가 옆으로 기울어진 상태에서 위에서 던지는 패턴입니다.
- 이 폼 자체가 바로 부상 위험이나 교정 대상이라고 보기는 어렵습니다.

### 그래서 무엇을 해야 하나?
- 폼을 크게 고치기보다는 어떤 타자에게 강하고 약한지를 먼저 확인하는 것이 더 중요합니다.
- 좌타자와 우타자 상대 성적이 다르면, 특정 타자 유형을 맡기는 방식으로 활용할 수 있습니다.
- 즉, Jackson은 "폼 교정 대상"이 아니라 "기용 방식 조정 대상"으로 보는 편이 적절합니다.

### 의사결정 포인트
- 폼 교정: 우선순위 낮음
- 매치업 분석: 우선순위 높음
- 권장 방향: 좌우 타자별 성적을 확인한 뒤, 강점이 있는 상황에 제한적으로 투입
"""
    show_pitcher_page("Jackson", "SV", "BS", interpretation, key_findings)
