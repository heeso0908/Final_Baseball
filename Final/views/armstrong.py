from shared import data
from views.pitcher import show_pitcher_page


def show():
    df = data['pitcher_ag']
    arm_df = df[df['player'] == 'Armstrong'].copy()
    top4 = arm_df.reindex(arm_df['cohens_d'].abs().sort_values(ascending=False).index)[:4]
    key_findings = [
        (row['label'], row['a_mean'], row['b_mean'], row['u_p'], row['cohens_d'])
        for _, row in top4.iterrows()
    ]
    interpretation = """
**불펜 투수 Armstrong — 세이브 vs 블론 세이브**

분석 결과 (Cohen's d 및 p-value 참고).

지표 차이의 정도와 통계적 유의성에 따라:

- **차이가 큰 지표**가 발견되면 → 폼 교정 영역
- **차이가 미미**하면 → Garcia처럼 외부 요인 의심

### 일반적 해석

불펜 투수는 짧은 등판이라 **워밍업 부족**으로 폼 변동이 클 수 있음.
일관성 향상을 위한 **루틴 표준화**가 도움이 될 가능성.
"""
    show_pitcher_page("Armstrong", "SV", "BS", interpretation, key_findings)