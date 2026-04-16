"""
투수/타자 유형 자동 분류 모듈

투수:
  - 투구손(우/좌): 손목 x좌표 이동량 비교
  - 팔 각도: 팔꿈치-어깨 y좌표 차이 기준
    overhand     : mean_diff < -0.05 (팔꿈치가 어깨보다 위)
    three_quarter: -0.05 ~ 0.02
    sidearm      : 0.02 ~ 0.08
    underhand    : 0.08 이상 (팔꿈치가 어깨보다 아래)

타자:
  - 타격손(우/좌): 손목 x이동 범위 비교
  - 스탠스: 초반 자세에서 양발 위치
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────

def get_pt(row, name):
    return (row[f"{name}_x"], row[f"{name}_y"])


def smooth(series, window=7):
    if len(series) < window:
        return np.array(series)
    return savgol_filter(series, window_length=window, polyorder=2)


# ──────────────────────────────────────────
# 1. 투수 유형 분류
# ──────────────────────────────────────────

# Statcast arm_angle 실측값 기반 슬롯 override
# arm_angle: overhand(<30°) / three_quarter(30~60°) / sidearm(60~80°) / underhand(>80°)
KNOWN_ARM_SLOTS = {
    "leiter": {"hand": "R", "arm_slot": "three_quarter", "arm_slot_angle": 45.5},
    "webb":   {"hand": "R", "arm_slot": "three_quarter", "arm_slot_angle": 34.1},
}


def classify_pitcher(pose_df: pd.DataFrame, fps: float, player_name: str = "") -> dict:
    """
    player_name: 소문자 이름 (예: "leiter") — KNOWN_ARM_SLOTS에 있으면 Statcast 실측값 사용
    """
    # 알려진 선수는 Statcast 값으로 override
    key = player_name.lower().strip()
    if key in KNOWN_ARM_SLOTS:
        known = KNOWN_ARM_SLOTS[key]
        return {
            "hand":              known["hand"],
            "arm_slot":          known["arm_slot"],
            "arm_slot_angle":    known["arm_slot_angle"],
            "arm_slot_y_diff":   None,
            "hand_confidence":   1.0,
            "x_factor_reliable": True,
            "note": (
                f"{'우' if known['hand'] == 'R' else '좌'}투 {known['arm_slot']} | "
                f"Statcast arm_angle {known['arm_slot_angle']}° (실측)"
            )
        }
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)

    # ── 투구손 감지: 더 많이 움직인 손이 투구손 ──
    rw_x = pd.Series(smooth(df["right_wrist_x"].values, window=5))
    lw_x = pd.Series(smooth(df["left_wrist_x"].values,  window=5))
    rw_range = float(rw_x.max() - rw_x.min())
    lw_range = float(lw_x.max() - lw_x.min())

    if rw_range > lw_range * 1.2:
        hand = "R"
        throw_el, throw_sh, throw_wr = "right_elbow", "right_shoulder", "right_wrist"
        hand_confidence = min(rw_range / (lw_range + 1e-6) / 2, 1.0)
    elif lw_range > rw_range * 1.2:
        hand = "L"
        throw_el, throw_sh, throw_wr = "left_elbow", "left_shoulder", "left_wrist"
        hand_confidence = min(lw_range / (rw_range + 1e-6) / 2, 1.0)
    else:
        hand = "R"
        throw_el, throw_sh, throw_wr = "right_elbow", "right_shoulder", "right_wrist"
        hand_confidence = 0.5

    # ── 팔 각도 감지: 릴리즈 구간 팔꿈치-어깨 y차이 ──
    n = len(df)
    release_zone = df[n * 2 // 3:]

    y_diffs = []
    for _, row in release_zone.iterrows():
        el_y = row[f"{throw_el}_y"]
        sh_y = row[f"{throw_sh}_y"]
        y_diffs.append(float(el_y - sh_y))

    mean_diff = float(np.median(y_diffs))

    # mean_diff < 0: 팔꿈치가 어깨보다 위 → 오버핸드
    # mean_diff > 0: 팔꿈치가 어깨보다 아래 → 사이드암/언더핸드
    if mean_diff < -0.05:
        arm_slot = "overhand"
        arm_slot_angle = round(abs(mean_diff) * 90, 1)
    elif mean_diff < 0.02:
        arm_slot = "three_quarter"
        arm_slot_angle = round(45 + mean_diff * 100, 1)
    elif mean_diff < 0.08:
        arm_slot = "sidearm"
        arm_slot_angle = round(75 + mean_diff * 100, 1)
    else:
        arm_slot = "underhand"
        arm_slot_angle = round(90 + mean_diff * 100, 1)

    x_factor_reliable = arm_slot in ["overhand", "three_quarter"]

    return {
        "hand":              hand,
        "arm_slot":          arm_slot,
        "arm_slot_angle":    arm_slot_angle,
        "arm_slot_y_diff":   round(mean_diff, 4),
        "hand_confidence":   round(hand_confidence, 2),
        "x_factor_reliable": x_factor_reliable,
        "note": (
            f"{'우' if hand == 'R' else '좌'}투 {arm_slot} | "
            f"y차이 {mean_diff:.3f} | "
            f"X-Factor 신뢰도: {'높음' if x_factor_reliable else '낮음'}"
        )
    }


# ──────────────────────────────────────────
# 2. 타자 유형 분류
# ──────────────────────────────────────────

def classify_batter(pose_df: pd.DataFrame, fps: float) -> dict:
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)
    n = len(df)

    # ── 타격손 감지: 스윙 구간 손목 이동 범위 ──
    swing_zone = df[n // 3: n * 2 // 3]
    if swing_zone.empty:
        swing_zone = df

    rw_x_range = float(swing_zone["right_wrist_x"].max() - swing_zone["right_wrist_x"].min())
    lw_x_range = float(swing_zone["left_wrist_x"].max()  - swing_zone["left_wrist_x"].min())

    if rw_x_range > lw_x_range * 1.2:
        hand = "R"
        front_ankle, back_ankle = "left_ankle", "right_ankle"
        hand_confidence = min(rw_x_range / (lw_x_range + 1e-6) / 2, 1.0)
    elif lw_x_range > rw_x_range * 1.2:
        hand = "L"
        front_ankle, back_ankle = "right_ankle", "left_ankle"
        hand_confidence = min(lw_x_range / (rw_x_range + 1e-6) / 2, 1.0)
    else:
        hand = "R"
        front_ankle, back_ankle = "left_ankle", "right_ankle"
        hand_confidence = 0.5

    # ── 스탠스 감지: 초반 자세에서 양발 위치 ──
    prep_zone = df[:n // 4]
    if prep_zone.empty:
        prep_zone = df.head(10)

    front_x = float(prep_zone[f"{front_ankle}_x"].mean())
    back_x  = float(prep_zone[f"{back_ankle}_x"].mean())
    front_y = float(prep_zone[f"{front_ankle}_y"].mean())
    back_y  = float(prep_zone[f"{back_ankle}_y"].mean())

    dx = front_x - back_x
    dy = front_y - back_y
    stance_angle = float(np.degrees(np.arctan2(dy, abs(dx) + 1e-6)))

    if stance_angle > 15:
        stance = "open"
    elif stance_angle < -15:
        stance = "closed"
    else:
        stance = "square"

    return {
        "hand":            hand,
        "stance":          stance,
        "stance_angle":    round(stance_angle, 1),
        "hand_confidence": round(hand_confidence, 2),
        "note": (
            f"{'우' if hand == 'R' else '좌'}타 | "
            f"스탠스: {stance} ({stance_angle:.1f}°) | "
            f"신뢰도: {hand_confidence:.0%}"
        )
    }


# ──────────────────────────────────────────
# 3. 유형별 기준값 조정
# ──────────────────────────────────────────

def get_adjusted_thresholds(pitcher_type: dict) -> dict:
    arm_slot = pitcher_type.get("arm_slot", "overhand")

    # OBP 실측 기반 기준값 (N=411, Driveline OpenBiomechanics)
    base = {
        "early_open_deg":                15.0,
        "knee_collapse_deg":            150.0,
        "arm_flyout_deg":               160.0,
        "x_factor_min_deg":              20.0,   # OBP 25%ile 기반
        "x_factor_max_deg":              45.0,   # OBP 75%ile 기반
        "x_factor_optimal":              32.0,   # OBP mean=32.3
        "shoulder_abduction_optimal":    87.0,   # OBP mean=86.7
        "shoulder_abduction_tolerance":  13.0,   # OBP IQR 기반
        "elbow_flexion_fp_min":          80.0,   # OBP 25%ile 기반
        "elbow_flexion_fp_max":         130.0,   # OBP 75%ile 기반
        "trunk_ahead_threshold":          0.05,
        "use_x_factor":                  True,
        "use_wrist_below_elbow":         True,
        "use_early_open":                True,
    }

    if arm_slot == "sidearm":
        base.update({
            "early_open_deg":        30.0,
            "use_x_factor":          False,
            "use_wrist_below_elbow": False,
            "arm_flyout_deg":        170.0,
        })
    elif arm_slot == "underhand":
        base.update({
            "early_open_deg":        45.0,
            "use_x_factor":          False,
            "use_wrist_below_elbow": False,
            "arm_flyout_deg":        180.0,
        })
    elif arm_slot == "three_quarter":
        base.update({
            "early_open_deg":   20.0,
            "x_factor_min_deg": 22.0,   # OBP 기반 조정
        })

    return base


# ──────────────────────────────────────────
# 4. 전체 분류 파이프라인
# ──────────────────────────────────────────

def classify_and_analyze(pose_df: pd.DataFrame, fps: float, mode: str = "pitch") -> dict:
    if mode == "pitch":
        pitcher_type = classify_pitcher(pose_df, fps)
        thresholds   = get_adjusted_thresholds(pitcher_type)

        print(f"\n[투수 유형 분류]")
        print(f"  {pitcher_type['note']}")
        print(f"  X-Factor 분석:      {'포함' if thresholds['use_x_factor'] else '제외'}")
        print(f"  팔올림 타이밍 분석: {'포함' if thresholds['use_wrist_below_elbow'] else '제외'}")
        print(f"  early_open 기준:    {thresholds['early_open_deg']}°")

        return {
            "type":       pitcher_type,
            "thresholds": thresholds,
            "hand":       pitcher_type["hand"],
        }
    else:
        batter_type = classify_batter(pose_df, fps)

        print(f"\n[타자 유형 분류]")
        print(f"  {batter_type['note']}")

        return {
            "type": batter_type,
            "hand": batter_type["hand"],
        }


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    import os

    CSV_PATH = "pose_output/hader_homerun_pose.csv"

    if not os.path.exists(CSV_PATH):
        print(f"파일 없음: {CSV_PATH}")
    else:
        pose_df = pd.read_csv(CSV_PATH)
        fps = 59.6

        print("=" * 60)
        print("투수 유형 자동 분류")
        print("=" * 60)
        result = classify_and_analyze(pose_df, fps, mode="pitch")

        print(f"\n투구손:          {result['type']['hand']}")
        print(f"팔 각도:          {result['type']['arm_slot']}")
        print(f"y차이(중앙값):   {result['type']['arm_slot_y_diff']}")
        print(f"X-Factor 포함:   {result['thresholds']['use_x_factor']}")
