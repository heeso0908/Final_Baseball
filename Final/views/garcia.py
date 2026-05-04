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
**마무리 투수 Garcia — 세이브 vs 블론 세이브, Null Finding**

대부분 지표에서 통계적 유의성 없음 (p > 0.05).

### 의미

- Garcia의 **폼 자체는 매우 일관**됨
- Save와 Blown Save 차이가 **폼에서 비롯되지 않음**

### 가능한 블론 세이브 원인 (모션 외 요인)

1. **구속 저하**: 누적 등판 피로
2. **구질 효과**: 변화구 구사율 변화
3. **매치업**: 특정 타자/타순에 약점
4. **상황적 압박**: 9회 vs 연장 등

### 시사점

- 폼 교정으로 블론 세이브 줄이기 **어려움**
- **Deployment 변경** (등판 상황 조절)
- **매치업 specialist 활용**: 특정 타자 회피
- 또는 **외부 영입** 검토

본인 분석에서 GM의 외부 클로저 영입 결정과 일치.
"""
    show_pitcher_page("Garcia", "SV", "BS", interpretation, key_findings)