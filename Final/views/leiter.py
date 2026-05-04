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
**선발 투수 Leiter — 삼진 vs 볼넷 분기**

분석 결과 통계적으로 강한 차이는 발견되지 않음 (대부분 p > 0.1).
이는 다음을 시사:

- **폼 자체는 일관**되게 유지됨
- 결과 차이는 폼이 아닌 **다른 요인**에서 비롯될 가능성
  - 구속/구질 변화
  - 제구점 (location)
  - 타자 매치업
  - 카운트 운영

**시사점**: 폼 교정보다 **피칭 디자인** 측면 검토 필요.
"""
    show_pitcher_page("Leiter", "SO", "Walk", interpretation, key_findings)