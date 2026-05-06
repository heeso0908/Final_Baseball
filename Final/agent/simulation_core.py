"""TEX 2025 residual-scenario simulation core.

This module mirrors the ML residual scenario logic in
`Final/5.통합본(0428+ML 수정)_v5.ipynb`.

Scope:
- It is not the full Markov game simulator from the notebook.
- It is the lightweight agent-facing residual engine: Ridge, Lasso,
  RandomForest, and XGBoost ensemble point estimates.
- Scenario uncertainty is reported as the standard deviation of the
  four model predictions (`pred_std`).

Important notebook parity points:
- Training data: 2015-2019 and 2021-2024 only.
- TEX 2025 baseline is reconstructed from the same raw files used in
  notebook Cell 26, not copied blindly from `mlb_team_seasons.csv`.
- Ridge/Lasso alpha search uses 60 log-spaced values and
  Leave-One-Season-Out GroupKFold.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_APP_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _APP_ROOT / "data_raw"

MODEL_FEATURES = [
    "sv_pct",
    "SV_pg",
    "onerun_wp",
    "xi_wp",
    "home_away_diff",
    "ERA",
    "WHIP",
    "k_bb",
    "K9",
    "BB9",
    "HR9",
    "ir_pct",
    "babip_against",
    "go_ao",
    "OPS",
    "rs_per_g",
    "sb_pct",
    "era_fip_diff",
]

LOWER_BETTER = {"ir_pct", "HR9", "BB9", "WHIP", "babip_against", "era_fip_diff"}

PYTH_CAPTURED = {"OPS", "rs_per_g", "ERA"}
RESIDUAL_PROXY_FEATS = [f for f in MODEL_FEATURES if f not in PYTH_CAPTURED]

_state: dict[str, Any] = {}


def _read_csv(name: str) -> pd.DataFrame:
    path = _DATA_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"required data file is missing: {path}")
    return pd.read_csv(path)


def _ip_float(value: Any) -> float:
    text = str(value)
    parts = text.split(".")
    if len(parts) == 2:
        return int(parts[0]) + int(parts[1]) / 3
    return float(text)


def _add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pyth_wp"] = df["RS"] ** 1.83 / (df["RS"] ** 1.83 + df["RA"] ** 1.83)
    df["pyth_W"] = (df["pyth_wp"] * df["G"]).round(1)
    df["residual"] = df["W"] - df["pyth_W"]
    df["home_away_diff"] = df["home_wp"] - df["away_wp"]
    df["k_bb"] = df["K9"] / df["BB9"].clip(lower=0.1)
    df["rs_per_g"] = df["RS"] / df["G"]
    df["SV_pg"] = df["SV"] / df["G"]
    return df


def _build_tex25_baseline(mlb_hist: pd.DataFrame) -> dict[str, float]:
    """Rebuild notebook Cell 26 TEX 2025 feature values."""
    game_log = _read_csv("texas_2025_game_log.csv")
    game_log["win"] = game_log["W/L"].astype(str).str.startswith("W")
    game_log["is_home"] = game_log["Home_Away"] == "Home"
    game_log["run_diff"] = game_log["R"] - game_log["RA"]
    game_log["onerun"] = game_log["run_diff"].abs() == 1
    game_log["extra"] = game_log["Inn"].fillna(9).astype(float) > 9

    sv_df = _read_csv("tex_2025_save_situation_splits.csv")
    team_sv = sv_df[sv_df["Name"] == "Team Total"].iloc[0]
    sv_val = float(team_sv["SV"])
    bs_val = float(team_sv["BS"])
    svo_val = sv_val + bs_val

    pit_logs = _read_csv("rangers_pitcher_gamelogs.csv")
    ip_total = pit_logs["IP"].apply(_ip_float).sum()
    # Notebook Cell 3 uses ER + 3 for the team ERA reconstruction.
    era_actual = (pit_logs["ER"].sum() + 3) * 9 / ip_total
    hr9_actual = pit_logs["HR"].sum() * 9 / ip_total
    k9_actual = pit_logs["SO"].sum() * 9 / ip_total
    bb9_actual = pit_logs["BB"].sum() * 9 / ip_total
    whip_actual = (pit_logs["H"].sum() + pit_logs["BB"].sum()) / ip_total

    tex_row = mlb_hist[(mlb_hist["team_id"] == 140) & (mlb_hist["year"] == 2025)].iloc[0]

    tex25: dict[str, float] = {
        "W": int(game_log["win"].sum()),
        "L": int((~game_log["win"]).sum()),
        "G": len(game_log),
        "RS": float(game_log["R"].sum()),
        "RA": float(game_log["RA"].sum()),
        "home_wp": float(game_log[game_log["is_home"]]["win"].mean()),
        "away_wp": float(game_log[~game_log["is_home"]]["win"].mean()),
        "onerun_wp": float(game_log[game_log["onerun"]]["win"].mean()),
        "xi_wp": float(game_log[game_log["extra"]]["win"].mean()) if game_log["extra"].any() else 0.5,
        "SV": sv_val,
        "BS": bs_val,
        "SVO": svo_val,
        "sv_pct": sv_val / svo_val if svo_val > 0 else float(tex_row["sv_pct"]),
        "ERA": round(float(era_actual), 2),
        "WHIP": round(float(whip_actual), 3),
        "K9": round(float(k9_actual), 2),
        "BB9": round(float(bb9_actual), 2),
        "HR9": round(float(hr9_actual), 2),
        "ir_pct": float(tex_row["ir_pct"]),
        "babip_against": float(tex_row["babip_against"]),
        "go_ao": float(tex_row["go_ao"]),
        "OPS": float(tex_row["OPS"]),
        "sb_pct": float(tex_row["sb_pct"]),
        "era_fip_diff": float(tex_row["era_fip_diff"]),
    }
    tex25["home_away_diff"] = tex25["home_wp"] - tex25["away_wp"]
    tex25["k_bb"] = tex25["K9"] / max(tex25["BB9"], 0.1)
    tex25["rs_per_g"] = round(tex25["RS"] / tex25["G"], 3)
    tex25["SV_pg"] = tex25["SV"] / tex25["G"]
    tex25["pyth_wp"] = tex25["RS"] ** 1.83 / (tex25["RS"] ** 1.83 + tex25["RA"] ** 1.83)
    tex25["pyth_W"] = round(tex25["pyth_wp"] * tex25["G"], 1)
    tex25["residual"] = round(tex25["W"] - tex25["pyth_W"], 1)
    return tex25


def _ensure_loaded() -> None:
    if "ridge" in _state:
        return

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import r2_score
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb

    raw = _read_csv("mlb_team_seasons.csv")
    full = _add_derived(raw)

    train = full[(full["year"] < 2025) & (full["year"] != 2020)].copy()
    train = train.dropna(subset=MODEL_FEATURES + ["residual"])

    X_raw = train[MODEL_FEATURES].values
    y = train["residual"].values
    groups = train["year"].values

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_raw)

    gkf = GroupKFold(n_splits=train["year"].nunique())
    gkf_splits = list(gkf.split(X_sc, y, groups))
    alphas = np.logspace(-2, 3, 60)

    ridge_scores = [
        cross_val_score(Ridge(alpha=a), X_sc, y, cv=gkf_splits, scoring="r2").mean()
        for a in alphas
    ]
    best_ridge_a = alphas[int(np.argmax(ridge_scores))]
    ridge = Ridge(alpha=best_ridge_a).fit(X_sc, y)

    lasso_scores = [
        cross_val_score(
            Lasso(alpha=a, max_iter=5000),
            X_sc,
            y,
            cv=gkf_splits,
            scoring="r2",
        ).mean()
        for a in alphas
    ]
    best_lasso_a = alphas[int(np.argmax(lasso_scores))]
    lasso = Lasso(alpha=best_lasso_a, max_iter=5000).fit(X_sc, y)

    rf = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=5,
        random_state=42,
    ).fit(X_raw, y)

    xgb_m = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight=3,
        random_state=42,
        verbosity=0,
    ).fit(X_raw, y)

    cv_models = [
        ("Ridge", make_pipeline(StandardScaler(), Ridge(alpha=ridge.alpha)), X_raw),
        ("Lasso", make_pipeline(StandardScaler(), Lasso(alpha=lasso.alpha, max_iter=5000)), X_raw),
        ("RF", rf, X_raw),
        ("XGBoost", xgb_m, X_raw),
    ]
    cv_mae = {
        name: float(
            -cross_val_score(model, X, y, cv=gkf_splits, scoring="neg_mean_absolute_error").mean()
        )
        for name, model, X in cv_models
    }
    cv_r2 = {}
    for name, model, X in cv_models:
        if name == "Ridge":
            train_r2 = r2_score(y, ridge.predict(X_sc))
        elif name == "Lasso":
            train_r2 = r2_score(y, lasso.predict(X_sc))
        else:
            train_r2 = r2_score(y, model.predict(X))
        scores = cross_val_score(model, X, y, cv=gkf_splits, scoring="r2")
        cv_r2[name] = {
            "train_r2": float(train_r2),
            "cv_r2_mean": float(scores.mean()),
            "cv_r2_std": float(scores.std()),
        }

    feat_std = train[MODEL_FEATURES].std()

    tex25 = _build_tex25_baseline(raw)
    tex_base = {f: float(tex25[f]) for f in MODEL_FEATURES}
    base_resid = _predict_ensemble(
        np.array([[tex_base[f] for f in MODEL_FEATURES]]),
        scaler,
        ridge,
        lasso,
        rf,
        xgb_m,
    )["mean"]

    base_pyth_w = float(tex25["pyth_W"])
    actual_w = int(tex25["W"])
    base_predicted_w_raw = base_pyth_w + base_resid
    # Agent-facing calibrated output keeps the baseline anchored to actual 81 W.
    calibration_offset = actual_w - base_predicted_w_raw

    _state.update(
        {
            "scaler": scaler,
            "ridge": ridge,
            "lasso": lasso,
            "rf": rf,
            "xgb": xgb_m,
            "feat_std": feat_std,
            "tex25": tex25,
            "tex_base": tex_base,
            "base_resid": base_resid,
            "base_pyth_w": base_pyth_w,
            "base_predicted_w_raw": base_predicted_w_raw,
            "tex_actual_w": actual_w,
            "tex_actual_residual": float(tex25["residual"]),
            "calibration_offset": calibration_offset,
            "best_ridge_alpha": float(best_ridge_a),
            "best_lasso_alpha": float(best_lasso_a),
            "cv_mae": cv_mae,
            "ensemble_cv_mae": float(sum(cv_mae.values()) / len(cv_mae)),
            "cv_r2": cv_r2,
        }
    )


def _predict_ensemble(vec_raw, scaler, ridge, lasso, rf, xgb_m) -> dict[str, Any]:
    vec_sc = scaler.transform(vec_raw)
    preds = np.array(
        [
            ridge.predict(vec_sc)[0],
            lasso.predict(vec_sc)[0],
            rf.predict(vec_raw)[0],
            xgb_m.predict(vec_raw)[0],
        ]
    )
    return {"mean": float(preds.mean()), "std": float(preds.std()), "preds": preds}


def _sigma_to_delta(feature: str, sigma: float) -> float:
    sign = -1 if feature in LOWER_BETTER else 1
    return sign * sigma * float(_state["feat_std"][feature])


def get_baseline() -> dict[str, Any]:
    """Return TEX 2025 actual values and the notebook-aligned ML baseline."""
    _ensure_loaded()
    return {
        "actual_W": _state["tex_actual_w"],
        "actual_residual": round(_state["tex_actual_residual"], 1),
        "pyth_W": _state["base_pyth_w"],
        "predicted_residual": round(_state["base_resid"], 3),
        "predicted_W_raw": round(_state["base_predicted_w_raw"], 1),
        "calibration_offset": round(_state["calibration_offset"], 2),
        "calibrated_baseline_W": float(_state["tex_actual_w"]),
        "ensemble_cv_mae": round(_state["ensemble_cv_mae"], 3),
        "model_cv_mae": {k: round(v, 3) for k, v in _state["cv_mae"].items()},
        "features": dict(_state["tex_base"]),
    }


def feature_info() -> dict[str, Any]:
    _ensure_loaded()
    return {
        "all_features": list(MODEL_FEATURES),
        "adjustable": list(RESIDUAL_PROXY_FEATS),
        "pyth_captured": sorted(PYTH_CAPTURED),
        "lower_better": sorted(LOWER_BETTER),
        "feature_std": {f: round(float(_state["feat_std"][f]), 4) for f in MODEL_FEATURES},
    }


def estimate_residual_scenario(sigmas: dict[str, float]) -> dict[str, Any]:
    """Estimate a residual scenario using signed sigma adjustments.

    `sigma > 0` always means "improvement". For lower-is-better features
    such as BB9, HR9, WHIP, and ir_pct, the raw feature value is decreased.
    """
    _ensure_loaded()

    warnings: list[str] = []
    valid_sigmas: dict[str, float] = {}
    for feature, sigma in sigmas.items():
        if feature not in MODEL_FEATURES:
            warnings.append(f"unknown feature: {feature}")
            continue
        if feature in PYTH_CAPTURED:
            warnings.append(
                f"{feature} is already captured by the pyth_W channel; residual-only adjustment is not recommended"
            )
        if abs(float(sigma)) > 1.5:
            warnings.append(f"{feature} sigma={float(sigma):+.2f} exceeds the recommended +/-1.5 range")
        valid_sigmas[feature] = float(sigma)

    adj_features = dict(_state["tex_base"])
    adjustments: dict[str, dict[str, float]] = {}
    for feature, sigma in valid_sigmas.items():
        delta_raw = _sigma_to_delta(feature, sigma)
        adj_features[feature] += delta_raw
        adjustments[feature] = {
            "sigma": round(sigma, 3),
            "delta_raw": round(delta_raw, 4),
            "new_value": round(adj_features[feature], 4),
        }

    if any(f in valid_sigmas for f in ("HR9", "BB9", "K9")):
        cfip_base = (
            (_state["tex_base"]["ERA"] - _state["tex_base"]["era_fip_diff"])
            - (
                13 * _state["tex_base"]["HR9"]
                - 2 * _state["tex_base"]["K9"]
                + 3 * _state["tex_base"]["BB9"]
            )
            / 9
        )
        new_fip = cfip_base + (
            13 * adj_features["HR9"] - 2 * adj_features["K9"] + 3 * adj_features["BB9"]
        ) / 9
        adj_features["era_fip_diff"] = adj_features["ERA"] - new_fip
        adjustments["era_fip_diff"] = {
            "sigma": 0.0,
            "delta_raw": round(adj_features["era_fip_diff"] - _state["tex_base"]["era_fip_diff"], 4),
            "new_value": round(adj_features["era_fip_diff"], 4),
        }

    vec_raw = np.array([[adj_features[f] for f in MODEL_FEATURES]])
    pred = _predict_ensemble(
        vec_raw,
        _state["scaler"],
        _state["ridge"],
        _state["lasso"],
        _state["rf"],
        _state["xgb"],
    )
    new_resid = pred["mean"]
    delta = new_resid - _state["base_resid"]
    predicted_w_raw = _state["base_pyth_w"] + new_resid
    predicted_w_calibrated = predicted_w_raw + _state["calibration_offset"]

    return {
        "predicted_W_calibrated": round(predicted_w_calibrated, 1),
        "predicted_W_raw": round(predicted_w_raw, 1),
        "predicted_residual": round(new_resid, 3),
        "delta": round(delta, 3),
        "pred_std": round(pred["std"], 4),
        "baseline_W_actual": _state["tex_actual_w"],
        "baseline_W_predicted_raw": round(_state["base_predicted_w_raw"], 1),
        "baseline_residual_predicted": round(_state["base_resid"], 3),
        "calibration_offset": round(_state["calibration_offset"], 2),
        "adjustments": adjustments,
        "warnings": warnings,
    }


simulate_scenario = estimate_residual_scenario


if __name__ == "__main__":
    baseline = get_baseline()
    print("=== Baseline ===")
    print(f"actual W:          {baseline['actual_W']}")
    print(f"pyth_W:            {baseline['pyth_W']}")
    print(f"actual residual:   {baseline['actual_residual']:+.1f}")
    print(f"ML residual:       {baseline['predicted_residual']:+.3f}")
    print(f"ML W raw:          {baseline['predicted_W_raw']}")
    print(f"calibration:       {baseline['calibration_offset']:+.2f}")

    print("\n=== Scenario: K9, BB9, HR9 +0.5 sigma improvement ===")
    result = estimate_residual_scenario({"K9": 0.5, "BB9": 0.5, "HR9": 0.5})
    print(f"calibrated W:      {result['predicted_W_calibrated']}")
    print(f"raw W:             {result['predicted_W_raw']}")
    print(f"delta:             {result['delta']:+.3f}")
    print(f"pred_std:          {result['pred_std']:.4f}")
