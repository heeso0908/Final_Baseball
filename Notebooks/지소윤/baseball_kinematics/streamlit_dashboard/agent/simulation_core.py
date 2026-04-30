"""TEX 2025 반사실 시뮬레이션 코어.

v3 노트북의 ML 잔차 앙상블(Ridge/Lasso/RF/XGB)을 모듈 단위로 추출.
챗봇의 simulate_scenario 도구가 이 모듈을 호출.

import 시 1회 학습(약 10초). 이후 호출은 즉시 응답.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Data 디렉토리 위치: streamlit_dashboard/agent/ → Final_Baseball/Data
_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATA_DIR = _REPO_ROOT / "Data"

MODEL_FEATURES = [
    'sv_pct', 'SV_pg', 'onerun_wp', 'xi_wp', 'home_away_diff',
    'ERA', 'WHIP', 'k_bb', 'K9', 'BB9', 'HR9',
    'ir_pct', 'babip_against', 'go_ao', 'OPS', 'rs_per_g',
    'sb_pct', 'era_fip_diff',
]

LOWER_BETTER = {'ir_pct', 'HR9', 'BB9', 'WHIP', 'babip_against', 'era_fip_diff'}

# pyth_W 채널과 중복되는 피처(시나리오 조정 제외 권장)
PYTH_CAPTURED = {'OPS', 'rs_per_g', 'ERA'}
RESIDUAL_PROXY_FEATS = [f for f in MODEL_FEATURES if f not in PYTH_CAPTURED]

_state: dict[str, Any] = {}


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['pyth_wp'] = df['RS']**1.83 / (df['RS']**1.83 + df['RA']**1.83)
    df['pyth_W'] = (df['pyth_wp'] * df['G']).round(1)
    df['residual'] = df['W'] - df['pyth_W']
    df['home_away_diff'] = df['home_wp'] - df['away_wp']
    df['k_bb'] = df['K9'] / df['BB9'].clip(lower=0.1)
    df['rs_per_g'] = df['RS'] / df['G']
    df['SV_pg'] = df['SV'] / df['G']
    return df


def _ensure_loaded() -> None:
    """모델·베이스라인을 1회 학습해 모듈 상태에 캐시."""
    if 'ridge' in _state:
        return

    from sklearn.linear_model import RidgeCV, LassoCV
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    csv_path = _DATA_DIR / "mlb_team_seasons.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"mlb_team_seasons.csv가 없습니다: {csv_path}")

    raw = pd.read_csv(csv_path)
    full = _add_derived(raw)

    # 학습: 2015~2019 + 2021~2024 (2020 제외, 2025 제외)
    train = full[(full['year'] < 2025) & (full['year'] != 2020)].copy()
    train = train.dropna(subset=MODEL_FEATURES + ['residual'])

    X_raw = train[MODEL_FEATURES].values
    y = train['residual'].values

    scaler = StandardScaler().fit(X_raw)
    X_sc = scaler.transform(X_raw)

    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0]).fit(X_sc, y)
    lasso = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0], max_iter=5000, cv=5).fit(X_sc, y)
    rf = RandomForestRegressor(
        n_estimators=300, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X_raw, y)
    xgb_m = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=2,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        random_state=42, verbosity=0,
    ).fit(X_raw, y)

    # 피처별 표준편차(σ → 원본 스케일 변환용)
    feat_std = train[MODEL_FEATURES].std()

    # TEX 2025 베이스라인
    tex_row = full[(full['team_id'] == 140) & (full['year'] == 2025)].iloc[0]
    tex_base = {f: float(tex_row[f]) for f in MODEL_FEATURES}
    base_resid = _predict(np.array([[tex_base[f] for f in MODEL_FEATURES]]),
                          scaler, ridge, lasso, rf, xgb_m)
    base_pyth_w = float(tex_row['pyth_W'])

    _state.update({
        'scaler': scaler, 'ridge': ridge, 'lasso': lasso, 'rf': rf, 'xgb': xgb_m,
        'feat_std': feat_std, 'tex_base': tex_base,
        'base_resid': base_resid, 'base_pyth_w': base_pyth_w,
        'tex_actual_w': int(tex_row['W']),
    })


def _predict(vec_raw, scaler, ridge, lasso, rf, xgb_m) -> float:
    vec_sc = scaler.transform(vec_raw)
    return float(np.mean([
        ridge.predict(vec_sc)[0],
        lasso.predict(vec_sc)[0],
        rf.predict(vec_raw)[0],
        xgb_m.predict(vec_raw)[0],
    ]))


def _sigma_to_delta(feature: str, sigma: float) -> float:
    """σ 단위 입력을 원본 스케일 변화량으로 변환. lower-better 피처는 부호 반전."""
    sign = -1 if feature in LOWER_BETTER else +1
    return sign * sigma * float(_state['feat_std'][feature])


def get_baseline() -> dict:
    """TEX 2025 실제 통계와 ML 베이스라인 잔차 예측."""
    _ensure_loaded()
    return {
        'actual_W': _state['tex_actual_w'],
        'pyth_W': _state['base_pyth_w'],
        'predicted_residual': round(_state['base_resid'], 3),
        'predicted_W': round(_state['base_pyth_w'] + _state['base_resid'], 1),
        'features': dict(_state['tex_base']),
    }


def feature_info() -> dict:
    """챗봇이 어떤 피처를 어느 방향으로 조정할지 판단할 때 사용."""
    _ensure_loaded()
    return {
        'all_features': list(MODEL_FEATURES),
        'adjustable': list(RESIDUAL_PROXY_FEATS),
        'pyth_captured': sorted(PYTH_CAPTURED),
        'lower_better': sorted(LOWER_BETTER),
        'feature_std': {f: round(float(_state['feat_std'][f]), 4) for f in MODEL_FEATURES},
    }


def simulate_scenario(sigmas: dict[str, float]) -> dict:
    """σ 단위 조정값을 받아 잔차/예상 승수 변화를 반환.

    Args:
        sigmas: {피처명: σ_배율} (예: {'K9': 0.3, 'BB9': -0.4})
                양수 = 개선 방향, 음수 = 악화 방향. lower-better 피처도 양수 = 개선.

    Returns:
        {
            'predicted_W': 예상 승수,
            'delta': 베이스라인 대비 잔차 변화,
            'baseline_W': TEX 2025 ML 베이스라인 예상 승수,
            'adjustments': 각 피처의 원본 스케일 변화량,
            'warnings': 검증 경고 (있을 때만),
        }
    """
    _ensure_loaded()

    warnings: list[str] = []
    valid_sigmas: dict[str, float] = {}
    for feature, sigma in sigmas.items():
        if feature not in MODEL_FEATURES:
            warnings.append(f"알 수 없는 피처: {feature}")
            continue
        if feature in PYTH_CAPTURED:
            warnings.append(
                f"{feature}는 pyth_W 채널과 중복 — 시나리오 조정에서 제외 권장"
            )
        if abs(sigma) > 1.5:
            warnings.append(f"{feature} σ={sigma:+.2f} → ±1.5 범위 초과 (외삽 주의)")
        valid_sigmas[feature] = float(sigma)

    # 피처 벡터 구성
    adj_features = dict(_state['tex_base'])
    adjustments = {}
    for feature, sigma in valid_sigmas.items():
        delta = _sigma_to_delta(feature, sigma)
        adj_features[feature] = adj_features[feature] + delta
        adjustments[feature] = {
            'sigma': round(sigma, 3),
            'delta_raw': round(delta, 4),
            'new_value': round(adj_features[feature], 4),
        }

    # HR9/BB9 변화 시 era_fip_diff 자동 재계산 (FIP 일관성)
    if any(f in valid_sigmas for f in ('HR9', 'BB9', 'K9')):
        cfip_base = ((_state['tex_base']['ERA'] - _state['tex_base']['era_fip_diff'])
                     - (13 * _state['tex_base']['HR9']
                        - 2 * _state['tex_base']['K9']
                        + 3 * _state['tex_base']['BB9']) / 9)
        new_fip = cfip_base + (13 * adj_features['HR9']
                               - 2 * adj_features['K9']
                               + 3 * adj_features['BB9']) / 9
        adj_features['era_fip_diff'] = adj_features['ERA'] - new_fip

    vec_raw = np.array([[adj_features[f] for f in MODEL_FEATURES]])
    new_resid = _predict(
        vec_raw, _state['scaler'], _state['ridge'], _state['lasso'],
        _state['rf'], _state['xgb'],
    )
    delta = new_resid - _state['base_resid']
    predicted_w = _state['base_pyth_w'] + new_resid
    baseline_w = _state['base_pyth_w'] + _state['base_resid']

    return {
        'predicted_W': round(predicted_w, 1),
        'delta': round(delta, 3),
        'baseline_W': round(baseline_w, 1),
        'baseline_residual': round(_state['base_resid'], 3),
        'adjustments': adjustments,
        'warnings': warnings,
    }


if __name__ == '__main__':
    # 빠른 sanity check
    print("=== 베이스라인 ===")
    bl = get_baseline()
    print(f"  실제 W: {bl['actual_W']}, pyth_W: {bl['pyth_W']}, "
          f"예측 W: {bl['predicted_W']} (residual {bl['predicted_residual']:+.2f})")

    print("\n=== 시나리오: K9 +0.3σ, BB9 -0.4σ ===")
    res = simulate_scenario({'K9': 0.3, 'BB9': -0.4})
    print(f"  예상 W: {res['predicted_W']} (delta {res['delta']:+.2f})")
    for f, a in res['adjustments'].items():
        print(f"    {f}: σ={a['sigma']:+.2f} → +{a['delta_raw']:+.4f}")

    print("\n=== 시나리오: 불펜 강화 (sv_pct +0.5σ, ir_pct +0.3σ) ===")
    res = simulate_scenario({'sv_pct': 0.5, 'ir_pct': 0.3})
    print(f"  예상 W: {res['predicted_W']} (delta {res['delta']:+.2f})")
