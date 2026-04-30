"""TEX 2025 반사실 시나리오 추정 코어 (ML 잔차 점추정).

v3 노트북 Cell 28의 잔차 앙상블(Ridge·Lasso·RF·XGB)을 모듈로 추출.
**Monte Carlo 시뮬레이션이 아니라 4모델 평균 점추정**임에 유의.
시나리오 불확실성은 `pred_std` (4모델 예측의 표준편차)로 표현한다.

v3 노트북과의 일치성:
- alpha 탐색: 60개 logspace + GroupKFold(year) — 노트북과 동일
- RandomForest: n_estimators=500, min_samples_leaf=5
- XGBoost: 동일 하이퍼파라미터
- 학습 데이터: 2015~2019 + 2021~2024 (2020 제외, 2025 제외)

베이스라인 보정:
- ML 예측 잔차(~-3.94)와 실제 잔차(-9.06) 차이를 calibration_offset로 노출.
- 모든 출력에 raw 예측값과 calibrated 값을 함께 제공.

import 시 1회 학습(약 20~30초). 이후 호출은 즉시 응답.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[5]
_DATA_DIR = _REPO_ROOT / "Data"

MODEL_FEATURES = [
    'sv_pct', 'SV_pg', 'onerun_wp', 'xi_wp', 'home_away_diff',
    'ERA', 'WHIP', 'k_bb', 'K9', 'BB9', 'HR9',
    'ir_pct', 'babip_against', 'go_ao', 'OPS', 'rs_per_g',
    'sb_pct', 'era_fip_diff',
]

LOWER_BETTER = {'ir_pct', 'HR9', 'BB9', 'WHIP', 'babip_against', 'era_fip_diff'}

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
    """모델·베이스라인을 1회 학습해 모듈 상태에 캐시.

    v3 노트북 Cell 28과 동일한 학습 방식:
    - alpha: np.logspace(-2, 3, 60), GroupKFold(year) R² 최대화
    - RF: n_estimators=500, XGB: max_depth=2, lr=0.03 등
    """
    if 'ridge' in _state:
        return

    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold, cross_val_score
    import xgboost as xgb

    csv_path = _DATA_DIR / "mlb_team_seasons.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"mlb_team_seasons.csv가 없습니다: {csv_path}")

    raw = pd.read_csv(csv_path)
    full = _add_derived(raw)

    train = full[(full['year'] < 2025) & (full['year'] != 2020)].copy()
    train = train.dropna(subset=MODEL_FEATURES + ['residual'])

    X_raw = train[MODEL_FEATURES].values
    y = train['residual'].values
    groups = train['year'].values

    scaler = StandardScaler().fit(X_raw)
    X_sc = scaler.transform(X_raw)

    # ── alpha 탐색: GroupKFold(year) + 60개 logspace (v3 노트북 동일) ─────────
    n_seasons = train['year'].nunique()
    gkf = GroupKFold(n_splits=n_seasons)
    gkf_splits = list(gkf.split(X_sc, y, groups))
    alphas = np.logspace(-2, 3, 60)

    ridge_scores = [
        cross_val_score(Ridge(alpha=a), X_sc, y, cv=gkf_splits, scoring='r2').mean()
        for a in alphas
    ]
    best_ridge_a = alphas[int(np.argmax(ridge_scores))]
    ridge = Ridge(alpha=best_ridge_a).fit(X_sc, y)

    lasso_scores = [
        cross_val_score(Lasso(alpha=a, max_iter=5000), X_sc, y,
                        cv=gkf_splits, scoring='r2').mean()
        for a in alphas
    ]
    best_lasso_a = alphas[int(np.argmax(lasso_scores))]
    lasso = Lasso(alpha=best_lasso_a, max_iter=5000).fit(X_sc, y)

    rf = RandomForestRegressor(
        n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1
    ).fit(X_raw, y)

    xgb_m = xgb.XGBRegressor(
        n_estimators=100, learning_rate=0.03, max_depth=2,
        subsample=0.7, colsample_bytree=0.7, min_child_weight=3,
        random_state=42, verbosity=0,
    ).fit(X_raw, y)

    feat_std = train[MODEL_FEATURES].std()

    # TEX 2025 베이스라인
    tex_row = full[(full['team_id'] == 140) & (full['year'] == 2025)].iloc[0]
    tex_base = {f: float(tex_row[f]) for f in MODEL_FEATURES}
    base_resid = _predict_ensemble(
        np.array([[tex_base[f] for f in MODEL_FEATURES]]),
        scaler, ridge, lasso, rf, xgb_m,
    )['mean']
    base_pyth_w = float(tex_row['pyth_W'])
    actual_w = int(tex_row['W'])

    # 보정 오프셋: ML 예측 베이스라인을 실제 W에 정렬
    base_predicted_w_raw = base_pyth_w + base_resid
    calibration_offset = actual_w - base_predicted_w_raw

    _state.update({
        'scaler': scaler, 'ridge': ridge, 'lasso': lasso, 'rf': rf, 'xgb': xgb_m,
        'feat_std': feat_std, 'tex_base': tex_base,
        'base_resid': base_resid,
        'base_pyth_w': base_pyth_w,
        'base_predicted_w_raw': base_predicted_w_raw,
        'tex_actual_w': actual_w,
        'calibration_offset': calibration_offset,
        'best_ridge_alpha': best_ridge_a,
        'best_lasso_alpha': best_lasso_a,
    })


def _predict_ensemble(vec_raw, scaler, ridge, lasso, rf, xgb_m) -> dict:
    """4모델 예측 → 평균 + std 동시 반환."""
    vec_sc = scaler.transform(vec_raw)
    preds = np.array([
        ridge.predict(vec_sc)[0],
        lasso.predict(vec_sc)[0],
        rf.predict(vec_raw)[0],
        xgb_m.predict(vec_raw)[0],
    ])
    return {'mean': float(preds.mean()), 'std': float(preds.std()), 'preds': preds}


def _sigma_to_delta(feature: str, sigma: float) -> float:
    """σ 단위 입력을 원본 스케일 변화량으로 변환.

    lower-better 피처(BB9, HR9 등)는 부호 반전하여
    **양수 sigma = 항상 개선 방향**으로 통일.
    """
    sign = -1 if feature in LOWER_BETTER else +1
    return sign * sigma * float(_state['feat_std'][feature])


def get_baseline() -> dict:
    """TEX 2025 실제 통계 + ML 베이스라인 + 보정값.

    Returns:
        - `actual_W`: 실제 승수 (81)
        - `pyth_W`: 피타고리안 기대 승수 (90.1)
        - `predicted_residual`: ML 예측 잔차 (~-3.94)
        - `predicted_W_raw`: ML 점추정 베이스라인 (~86.2)
        - `calibration_offset`: 실제-예측 차이 (~-5.2)
        - `calibrated_baseline_W`: 보정된 베이스라인 = actual_W
        - `features`: TEX 2025 18개 피처 값
    """
    _ensure_loaded()
    return {
        'actual_W': _state['tex_actual_w'],
        'pyth_W': _state['base_pyth_w'],
        'predicted_residual': round(_state['base_resid'], 3),
        'predicted_W_raw': round(_state['base_predicted_w_raw'], 1),
        'calibration_offset': round(_state['calibration_offset'], 2),
        'calibrated_baseline_W': float(_state['tex_actual_w']),
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


def estimate_residual_scenario(sigmas: dict[str, float]) -> dict:
    """σ 조정값을 받아 ML 잔차 점추정 (Monte Carlo 아님).

    4모델(Ridge·Lasso·RF·XGB) 예측의 평균을 반환하며, 시나리오의
    **불확실성은 4모델 std**(`pred_std`)로 정량화한다.

    Args:
        sigmas: {피처명: σ_배율}, 양수=개선, 음수=악화.
                lower-better 피처(BB9 등)도 양수=개선으로 통일.

    Returns:
        - `predicted_W_raw`: 보정 전 ML 예측 승수
        - `predicted_W_calibrated`: actual_W 기준 보정값 (사용자에게 보여줄 값)
        - `delta`: 베이스라인 잔차 대비 변화 (보정 전후 동일)
        - `pred_std`: 4모델 예측 표준편차 (시나리오 불확실성)
        - `baseline_W_actual`: TEX 2025 실제 81승
        - `baseline_W_predicted_raw`: ML 베이스라인 (~86.2)
        - `calibration_offset`: 보정 오프셋
        - `adjustments`: 피처별 σ → 원본 스케일 변화량 + new_value
        - `warnings`: 입력 검증 경고
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

    adj_features = dict(_state['tex_base'])
    adjustments = {}
    for feature, sigma in valid_sigmas.items():
        delta_raw = _sigma_to_delta(feature, sigma)
        adj_features[feature] = adj_features[feature] + delta_raw
        adjustments[feature] = {
            'sigma': round(sigma, 3),
            'delta_raw': round(delta_raw, 4),
            'new_value': round(adj_features[feature], 4),
        }

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
    pred = _predict_ensemble(
        vec_raw, _state['scaler'], _state['ridge'], _state['lasso'],
        _state['rf'], _state['xgb'],
    )
    new_resid = pred['mean']
    delta = new_resid - _state['base_resid']
    predicted_w_raw = _state['base_pyth_w'] + new_resid
    predicted_w_calibrated = predicted_w_raw + _state['calibration_offset']

    return {
        'predicted_W_calibrated': round(predicted_w_calibrated, 1),
        'predicted_W_raw': round(predicted_w_raw, 1),
        'delta': round(delta, 3),
        'pred_std': round(pred['std'], 4),
        'baseline_W_actual': _state['tex_actual_w'],
        'baseline_W_predicted_raw': round(_state['base_predicted_w_raw'], 1),
        'calibration_offset': round(_state['calibration_offset'], 2),
        'adjustments': adjustments,
        'warnings': warnings,
    }


# 하위 호환 alias (기존 코드/외부 호출이 깨지지 않도록 유지)
simulate_scenario = estimate_residual_scenario


if __name__ == '__main__':
    print("=== 베이스라인 ===")
    bl = get_baseline()
    print(f"  실제 W:           {bl['actual_W']}")
    print(f"  pyth_W:           {bl['pyth_W']}")
    print(f"  ML 예측 잔차:     {bl['predicted_residual']:+.2f}")
    print(f"  ML 예측 W (raw):  {bl['predicted_W_raw']}")
    print(f"  보정 오프셋:      {bl['calibration_offset']:+.2f}")
    print(f"  보정 베이스라인:  {bl['calibrated_baseline_W']}")

    print("\n=== 시나리오: K9 +0.5σ, BB9 +0.5σ, HR9 +0.5σ (전부 개선) ===")
    res = estimate_residual_scenario({'K9': 0.5, 'BB9': 0.5, 'HR9': 0.5})
    print(f"  보정 W:    {res['predicted_W_calibrated']}")
    print(f"  raw W:     {res['predicted_W_raw']}")
    print(f"  delta:     {res['delta']:+.2f}")
    print(f"  pred_std:  {res['pred_std']:.4f}")

    print("\n=== 시나리오: 강한 전반 개선 (대부분 +0.7σ) ===")
    res = estimate_residual_scenario({
        'K9': 0.7, 'BB9': 0.7, 'HR9': 0.7,
        'sv_pct': 0.7, 'ir_pct': 0.7, 'onerun_wp': 0.5,
    })
    print(f"  보정 W:    {res['predicted_W_calibrated']}")
    print(f"  delta:     {res['delta']:+.2f}, pred_std: {res['pred_std']:.4f}")
