from shared import data
from views.pitcher import show_pitcher_page


def show():
    df = data['pitcher_ag']
    garcia_df = df[df['player'] == 'Garcia'].copy()
    top4 = garcia_df.reindex(garcia_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    interpretation = """
**마무리 투수 Garcia - 세이브 vs 블론 세이브 해석**

### 한 줄 결론
Garcia의 블론 세이브는 투구폼 문제보다는 기용 상황과 매치업 문제일 가능성이 큽니다.

### 쉽게 풀어보면
- 세이브 성공 상황과 블론 세이브 상황에서 몸 움직임 차이가 크게 보이지 않았습니다.
- 즉, 블론 세이브가 났다고 해서 그날 폼이 확실히 달랐다고 말하기는 어렵습니다.
- 이런 경우에는 상대 타자, 연투 여부, 구종 조합, 공의 위치 같은 외부 요인을 더 봐야 합니다.

### 그래서 무엇을 해야 하나?
- 폼을 고치기보다, 어떤 상황에서 Garcia를 투입할지 다시 설계하는 것이 중요합니다.
- 특정 타순이나 특정 유형의 타자에게 약하다면, 마무리 고정보다는 상황별 기용이 더 나을 수 있습니다.

### 의사결정 포인트
- 폼 교정: 우선순위 낮음
- 기용 방식: 우선 검토
- 권장 방향: 상대 타자 유형, 연투 여부, 세이브 상황의 타순을 기준으로 배치 조정
"""
    show_pitcher_page("Garcia", "SV", "BS", interpretation, key_findings)
