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
**불펜 투수 Armstrong - 세이브 vs 블론 세이브 해석**

### 한 줄 결론
Armstrong은 비교 기준에 가까운 선수이며, 큰 폼 문제를 단정하기 어렵습니다.

### 쉽게 풀어보면
- 일부 지표에서 차이는 보이지만, 표본이 작아서 확실한 결론으로 보기에는 조심스럽습니다.
- 불펜 투수는 등판 간격, 몸을 푸는 시간, 연투 여부에 따라 움직임이 흔들릴 수 있습니다.
- 따라서 특정 동작을 바로 고치기보다, 등판 전 준비 과정과 피로도를 함께 봐야 합니다.

### 그래서 무엇을 해야 하나?
- Armstrong은 폼 교정 대상이라기보다 불펜 운영의 기준점으로 활용하는 편이 좋습니다.
- 좋은 날과 흔들린 날의 차이를 루틴, 연투, 상대 타자 유형과 연결해서 확인해야 합니다.

### 의사결정 포인트
- 폼 교정: 보조적으로 검토
- 루틴 관리: 우선 검토
- 권장 방향: 등판 전 준비 시간, 연투 후 성적, 상대 타자 유형별 성적을 함께 점검
"""
    show_pitcher_page("Armstrong", "SV", "BS", interpretation, key_findings)
