"""TEX 2025 시나리오 시뮬레이터 — app_260501.py 연동용.

simulation_core(ML 잔차 앙상블)를 6개 사전정의 시나리오에 연결한다.
Monte Carlo는 pred_std 기반 정규분포 샘플링으로 근사한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# simulation_core 위치: Final/agent/simulation_core.py
_AGENT_DIR = str(Path(__file__).resolve().parents[2] / "Final" / "agent")
if _AGENT_DIR not in sys.path:
    sys.path.insert(0, _AGENT_DIR)

import simulation_core  # noqa: E402

# ── 시나리오 → σ 조정 매핑 ────────────────────────────────────────────────────
# 양수 σ = 항상 개선 방향 (lower-better 피처도 동일 규칙)
SCENARIO_SIGMAS: dict[str, dict[str, float]] = {
    "Baseline 2025": {},
    "Bullpen Upgrade": {
        "sv_pct":    0.50,
        "ir_pct":    0.50,
        "onerun_wp": 0.30,
    },
    "Rotation Spike": {
        "K9":   0.30,
        "BB9":  0.30,
        "HR9":  0.30,
        "WHIP": 0.30,
    },
    "Langford Leap": {
        "onerun_wp": 0.30,
        "xi_wp":     0.30,
        "sb_pct":    0.20,
    },
    "Hopeful Composite": {
        "sv_pct":    0.70,
        "ir_pct":    0.50,
        "K9":        0.50,
        "BB9":       0.50,
        "HR9":       0.50,
        "onerun_wp": 0.40,
    },
    "Risk Case": {
        "sv_pct":    -0.50,
        "ir_pct":    -0.30,
        "K9":        -0.30,
        "BB9":       -0.30,
        "onerun_wp": -0.40,
    },
}


def run_simulation(raw_dir: str, scenario_name: str, n_sims: int = 200) -> dict:
    """지정 시나리오로 ML 잔차 점추정 + Monte Carlo 근사.

    Args:
        raw_dir: 데이터 디렉토리 경로 (현재는 simulation_core 내부에서 처리).
        scenario_name: SCENARIO_SIGMAS 키 중 하나.
        n_sims: 샘플링 횟수 (pred_std 기반 정규분포).

    Returns:
        mean_W, median_W, std_W, p5_W, p95_W, p_90plus_pct,
        delta, pred_std, calibrated_W, scenario.
    """
    sigmas = SCENARIO_SIGMAS.get(scenario_name, {})
    result = simulation_core.estimate_residual_scenario(sigmas)

    mean_w = result["predicted_W_calibrated"]
    pred_std = result["pred_std"]

    rng = np.random.default_rng(42)
    samples = rng.normal(mean_w, max(pred_std, 0.5), size=n_sims)

    return {
        "scenario": scenario_name,
        "mean_W": round(float(samples.mean()), 1),
        "median_W": round(float(np.median(samples)), 1),
        "std_W": round(float(samples.std()), 2),
        "p5_W": round(float(np.percentile(samples, 5)), 1),
        "p95_W": round(float(np.percentile(samples, 95)), 1),
        "p_90plus_pct": round(float((samples >= 90).mean() * 100), 1),
        "delta": result["delta"],
        "pred_std": pred_std,
        "calibrated_W": mean_w,
        "adjustments": result.get("adjustments", {}),
        "warnings": result.get("warnings", []),
    }


def build_scenario_snapshots(raw_dir: str) -> dict:
    """6개 시나리오 전부 실행해 결과 dict 반환.

    Args:
        raw_dir: 데이터 디렉토리 경로.

    Returns:
        {scenario_name: run_simulation 결과} 형태의 dict.
    """
    return {
        name: run_simulation(raw_dir, name)
        for name in SCENARIO_SIGMAS
    }
