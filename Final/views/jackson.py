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
**불펜 투수 Jackson — 세이브 vs 블론 세이브, 해석**

**해석**: **Lateral Trunk Tilt Overhand** 수치만 보면 Sidearm처럼 보이지만,
실제로는 측면 굴곡이 있는 오버핸드 패턴으로 해석됨.
- Trunk 3D / Trunk XZ ratio 1.2-1.3 → 측면 굴곡 magnitude
- Walker Buehler, Clay Holmes archetype에 가까움

### 시사점 — Form Correction 아닌 Deployment

폼은 그 자체로 **건강한 패턴** (sidearm처럼 부상 위험 X).
다만 좌타자/우타자 매치업 특성이 다를 수 있음.

**정책 함의**:
- 폼 교정 X
- **Specialist Deployment**: 특정 매치업에서 활용
- **Platoon split** 분석 → 강점 매치업 식별
"""
    show_pitcher_page("Jackson", "SV", "BS", interpretation, key_findings)