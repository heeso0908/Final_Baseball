"""
야구 키네마틱 분석 모듈
- classifier.py의 thresholds를 받아서 유형별 기준값 적용
- 사이드암/언더핸드 전용 항목 자동 제외
- X-Factor 각도 정규화 수정
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


# ──────────────────────────────────────────
# 기본 기준값 (classifier에서 덮어씀)
# ──────────────────────────────────────────

DEFAULT_PITCH_THRESHOLDS = {
    # ── X-Factor (힙-어깨 분리) ──────────────────────────────
    # OBP 실측: mean=32.3°, 25~75%ile=27.6~36.7°
    # 기존 논문 기준(35~60°)보다 낮음 → OBP 기준으로 수정
    "x_factor_min_deg":              20.0,   # 이 이하면 분리 부족
    "x_factor_max_deg":              45.0,   # 이 이상이면 과도한 분리
    "x_factor_optimal":              32.0,   # OBP 중앙값

    # ── 팔꿈치 ─────────────────────────────────────────────
    # OBP 실측: 풋플랜트 시 굴곡 mean=103°, 25~75%ile=92~116°
    # arm_flyout: 팔꿈치 신전 각도 (굴곡의 반대) > 160° = 비효율
    "arm_flyout_deg":               160.0,
    "elbow_flexion_fp_min":          80.0,   # 풋플랜트 시 팔꿈치 굴곡 최소
    "elbow_flexion_fp_max":         130.0,   # 풋플랜트 시 팔꿈치 굴곡 최대

    # ── 어깨 ────────────────────────────────────────────────
    # OBP 실측: 어깨 외전 mean=86.7°, 25~75%ile=80.3~93.3°
    "shoulder_abduction_optimal":    87.0,   # OBP 평균값으로 수정 (기존 90°)
    "shoulder_abduction_tolerance":  13.0,   # ±13° (OBP IQR 기반)

    # ── 몸통/골반 ───────────────────────────────────────────
    # OBP 실측: 몸통 각속도 mean=1055°/s, 골반 751°/s
    # 2D에서는 절대값 신뢰 낮음 → 상대 순서만 사용
    "trunk_rotational_velo_ref":   1055.0,   # 참고용 (검증 미사용)
    "pelvis_rotational_velo_ref":   752.0,   # 참고용 (검증 미사용)

    # ── 무릎 ────────────────────────────────────────────────
    # 논문 기준 유지 (OBP에 무릎 각도 기준값 없음)
    "knee_collapse_deg":            150.0,

    # ── early_open (몸 일찍 열림) ───────────────────────────
    # OBP X-Factor 데이터에서 역산: 풋플랜트 전 분리각 > 15° = 비효율
    "early_open_deg":                15.0,

    # ── 기타 ────────────────────────────────────────────────
    "trunk_ahead_threshold":          0.05,
    "use_x_factor":                  True,
    "use_wrist_below_elbow":         True,
    "use_early_open":                True,

    # ── 출처 ────────────────────────────────────────────────
    # Driveline OpenBiomechanics Project (OBP)
    # N=411 투구, 대학~마이너 수준 우완 투수
    # https://github.com/drivelineresearch/openbiomechanics
}

DEFAULT_BAT_THRESHOLDS = {
    "trunk_ahead_threshold":  0.05,
    "hip_instability_std":    0.05,
    "hip_straight_deg":      150.0,
    "knee_dominant_deg":     120.0,
    "sway_threshold":          0.08,
}


# ──────────────────────────────────────────
# 유틸
# ──────────────────────────────────────────

def calc_angle(a, b, c) -> float:
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))


def rotation_angle_2d(p_left, p_right) -> float:
    return float(np.degrees(np.arctan2(
        p_right[1] - p_left[1],
        p_right[0] - p_left[0]
    )))


def angle_diff_normalized(a: float, b: float) -> float:
    """두 각도 차이를 -180~180° 범위로 정규화"""
    diff = a - b
    while diff > 180:  diff -= 360
    while diff < -180: diff += 360
    return diff


def get_pt(row, name):
    return (row[f"{name}_x"], row[f"{name}_y"])


def smooth(series, window=7):
    if len(series) < window:
        return np.array(series)
    return savgol_filter(series, window_length=window, polyorder=2)


# ──────────────────────────────────────────
# 1. 이벤트 자동 감지
# ──────────────────────────────────────────

def detect_pitch_events(pose_df: pd.DataFrame, fps: float, pitcher_hand: str = "R") -> dict:
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)
    front_ankle = "left_ankle"  if pitcher_hand == "R" else "right_ankle"
    throw_wrist = "right_wrist" if pitcher_hand == "R" else "left_wrist"

    ankle_y = pd.Series(smooth(df[f"{front_ankle}_y"].values, window=9))
    mid = len(ankle_y) // 3
    fp_local_idx = int(ankle_y[mid:].idxmax())
    fp_frame = int(df.loc[fp_local_idx, "frame"])
    fp_sec   = float(df.loc[fp_local_idx, "time_sec"])

    wrist_x  = pd.Series(smooth(df[f"{throw_wrist}_x"].values, window=5))
    wrist_vx = np.gradient(wrist_x.values, 1.0 / fps)

    fp_idx = df[df["frame"] == fp_frame].index
    fp_idx2 = fp_idx[0] if len(fp_idx) > 0 else mid

    after_fp_vx = np.abs(wrist_vx[fp_idx2:])
    rel_local_idx = fp_idx2 + int(np.argmax(after_fp_vx)) if len(after_fp_vx) > 0 else fp_idx2

    rel_frame = int(df.loc[rel_local_idx, "frame"])
    rel_sec   = float(df.loc[rel_local_idx, "time_sec"])
    valid = fp_frame < rel_frame

    print(f"  [투수 이벤트] 풋 플랜트: {fp_sec:.2f}초 (frame {fp_frame}) | 릴리즈: {rel_sec:.2f}초 (frame {rel_frame})")
    if not valid:
        print("  ⚠ 이벤트 순서 이상 - 수동 확인 필요")

    return {
        "foot_plant_frame": fp_frame, "foot_plant_sec": fp_sec,
        "release_frame":    rel_frame, "release_sec":   rel_sec,
        "valid":            valid,
    }


def detect_batting_events(pose_df: pd.DataFrame, fps: float, batter_hand: str = "R") -> dict:
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)
    front_ankle = "left_ankle"  if batter_hand == "R" else "right_ankle"
    swing_wrist = "right_wrist" if batter_hand == "R" else "left_wrist"

    hip_cx = (df["left_hip_x"] + df["right_hip_x"]) / 2
    hip_cx_smooth = pd.Series(smooth(hip_cx.values, window=9))
    mid = len(hip_cx_smooth) // 2
    load_idx = int(hip_cx_smooth[:mid].idxmax() if batter_hand == "R" else hip_cx_smooth[:mid].idxmin())
    load_frame = int(df.loc[load_idx, "frame"])
    load_sec   = float(df.loc[load_idx, "time_sec"])

    ankle_y  = pd.Series(smooth(df[f"{front_ankle}_y"].values, window=9))
    fp_idx   = int(ankle_y[load_idx:].idxmax())
    fp_frame = int(df.loc[fp_idx, "frame"])
    fp_sec   = float(df.loc[fp_idx, "time_sec"])

    wrist_x  = pd.Series(smooth(df[f"{swing_wrist}_x"].values, window=5))
    wrist_vx = np.abs(np.gradient(wrist_x.values, 1.0 / fps))
    after_fp  = wrist_vx[fp_idx:]
    contact_idx   = fp_idx + int(np.argmax(after_fp)) if len(after_fp) > 0 else fp_idx
    contact_frame = int(df.loc[contact_idx, "frame"])
    contact_sec   = float(df.loc[contact_idx, "time_sec"])

    valid = load_frame < fp_frame < contact_frame
    print(f"  [타자 이벤트] 로드: {load_sec:.2f}초 | 풋플랜트: {fp_sec:.2f}초 | 컨택: {contact_sec:.2f}초")
    if not valid:
        print("  ⚠ 이벤트 순서 이상")

    return {
        "load_frame": load_frame, "load_sec": load_sec,
        "foot_plant_frame": fp_frame, "foot_plant_sec": fp_sec,
        "contact_frame": contact_frame, "contact_sec": contact_sec,
        "valid": valid,
    }


# ──────────────────────────────────────────
# 2. 투구 비효율 감지 (thresholds 연동)
# ──────────────────────────────────────────

def detect_pitching_inefficiency(
    pose_df: pd.DataFrame,
    fps: float,
    pitcher_hand: str = "R",
    events: dict = None,
    thresholds: dict = None,  # ← classifier에서 받은 기준값
) -> dict:

    t = thresholds if thresholds else DEFAULT_PITCH_THRESHOLDS
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)

    if events is None:
        events = detect_pitch_events(pose_df, fps, pitcher_hand)

    fp_frame  = events["foot_plant_frame"]
    rel_frame = events["release_frame"]

    if pitcher_hand == "R":
        throw_sh, throw_el, throw_wr = "right_shoulder", "right_elbow", "right_wrist"
        front_hip, front_knee, front_ank = "left_hip", "left_knee", "left_ankle"
    else:
        throw_sh, throw_el, throw_wr = "left_shoulder", "left_elbow", "left_wrist"
        front_hip, front_knee, front_ank = "right_hip", "right_knee", "right_ankle"

    pre_fp    = df[df["frame"] <= fp_frame]
    fp_row    = df[df["frame"] == fp_frame]
    fp_to_rel = df[(df["frame"] >= fp_frame) & (df["frame"] <= rel_frame)]
    rel_row   = df[df["frame"] == rel_frame]

    results = {}

    # 1. Sway: 시작 ~ 풋 플랜트
    if not pre_fp.empty:
        hip_cx = (pre_fp["left_hip_x"] + pre_fp["right_hip_x"]) / 2
        sway_range = float(hip_cx.max() - hip_cx.min())
        results["sway_range"]    = round(sway_range, 4)
        results["sway_detected"] = sway_range > 0.06

    # 2. 몸 일찍 열림: 시작 ~ 풋 플랜트 (use_early_open 체크)
    if t.get("use_early_open", True):
        early_open_frames = []
        for _, row in pre_fp.iterrows():
            sh_rot  = rotation_angle_2d(get_pt(row, "left_shoulder"), get_pt(row, "right_shoulder"))
            hip_rot = rotation_angle_2d(get_pt(row, "left_hip"),      get_pt(row, "right_hip"))
            sep = angle_diff_normalized(sh_rot, hip_rot)
            early_open_frames.append(sep > t["early_open_deg"])
        results["early_open_pct"] = round(float(np.mean(early_open_frames)), 3) if early_open_frames else None
    else:
        results["early_open_pct"] = None
        results["early_open_skipped"] = "사이드암/언더핸드 제외"

    # 3. 팔 올림 타이밍: 풋 플랜트 시점 (use_wrist_below_elbow 체크)
    if t.get("use_wrist_below_elbow", True):
        if not fp_row.empty:
            row = fp_row.iloc[0]
            results["wrist_below_elbow_at_fp"] = bool(
                get_pt(row, throw_wr)[1] > get_pt(row, throw_el)[1]
            )
    else:
        results["wrist_below_elbow_at_fp"] = None
        results["wrist_below_elbow_skipped"] = "사이드암/언더핸드 제외"

    # 4. 몸통 일찍 나감: 풋 플랜트 시점
    if not fp_row.empty:
        row = fp_row.iloc[0]
        sh_cx  = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
        hip_cx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
        results["trunk_ahead_at_fp"] = bool(sh_cx < hip_cx - t["trunk_ahead_threshold"])

    # 5. 무릎 무너짐: 풋 플랜트 ~ 릴리즈
    knee_collapse_frames = []
    for _, row in fp_to_rel.iterrows():
        angle = calc_angle(get_pt(row, front_hip), get_pt(row, front_knee), get_pt(row, front_ank))
        knee_collapse_frames.append(angle < t["knee_collapse_deg"])
    results["knee_collapse_pct"] = round(float(np.mean(knee_collapse_frames)), 3) if knee_collapse_frames else None

    # 6. 팔 벌어짐: 시작 ~ 풋 플랜트
    arm_flyout_frames = []
    for _, row in pre_fp.iterrows():
        angle = calc_angle(get_pt(row, throw_sh), get_pt(row, throw_el), get_pt(row, throw_wr))
        arm_flyout_frames.append(angle > t["arm_flyout_deg"])
    results["arm_flyout_pct"] = round(float(np.mean(arm_flyout_frames)), 3) if arm_flyout_frames else None

    # 7. 공 놓는 타이밍: 릴리즈 시점
    if not rel_row.empty:
        row = rel_row.iloc[0]
        wr_x  = row[f"{throw_wr}_x"]
        ank_x = row[f"{front_ank}_x"]
        results["late_release"] = bool(wr_x > ank_x) if pitcher_hand == "R" else bool(wr_x < ank_x)

    # 8. X-Factor: 시작 ~ 풋 플랜트 (use_x_factor 체크)
    if t.get("use_x_factor", True):
        xf_vals = []
        for _, row in pre_fp.iterrows():
            sh_rot  = rotation_angle_2d(get_pt(row, "left_shoulder"), get_pt(row, "right_shoulder"))
            hip_rot = rotation_angle_2d(get_pt(row, "left_hip"),      get_pt(row, "right_hip"))
            xf_vals.append(abs(angle_diff_normalized(sh_rot, hip_rot)))
        if xf_vals:
            max_xf = max(xf_vals)
            results["x_factor_max"] = round(max_xf, 2)
            results["x_factor_status"] = (
                "optimal"  if t["x_factor_min_deg"] <= max_xf <= t["x_factor_max_deg"]
                else "too_low"  if max_xf < t["x_factor_min_deg"]
                else "too_high"
            )
    else:
        results["x_factor_max"]    = None
        results["x_factor_status"] = "skipped (사이드암/언더핸드)"

    results["events"] = events
    return results


# ──────────────────────────────────────────
# 3. 타격 비효율 감지
# ──────────────────────────────────────────

def detect_batting_inefficiency(
    pose_df: pd.DataFrame,
    fps: float,
    batter_hand: str = "R",
    events: dict = None,
) -> dict:
    t = DEFAULT_BAT_THRESHOLDS
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)

    if events is None:
        events = detect_batting_events(pose_df, fps, batter_hand)

    load_frame    = events["load_frame"]
    fp_frame      = events["foot_plant_frame"]
    contact_frame = events["contact_frame"]

    if batter_hand == "R":
        front_hip, front_knee, front_ank = "left_hip",  "left_knee",  "left_ankle"
        back_hip,  back_knee,  back_ank  = "right_hip", "right_knee", "right_ankle"
        back_sh = "right_shoulder"
    else:
        front_hip, front_knee, front_ank = "right_hip", "right_knee", "right_ankle"
        back_hip,  back_knee,  back_ank  = "left_hip",  "left_knee",  "left_ankle"
        back_sh = "left_shoulder"

    load_to_fp  = df[(df["frame"] >= load_frame) & (df["frame"] <= fp_frame)]
    contact_row = df[df["frame"] == contact_frame]

    results = {}

    # 1. Sway: 로드 ~ 풋 플랜트
    if not load_to_fp.empty:
        hip_cx = (load_to_fp["left_hip_x"] + load_to_fp["right_hip_x"]) / 2
        sway_range = float(hip_cx.max() - hip_cx.min())
        results["sway_range"]    = round(sway_range, 4)
        results["sway_detected"] = sway_range > t["sway_threshold"]

    # 2. 무릎 위주 힘 모음: 로드 ~ 풋 플랜트
    knee_dominant_frames = []
    for _, row in load_to_fp.iterrows():
        b_sh  = get_pt(row, back_sh)
        b_hip = get_pt(row, back_hip)
        b_kn  = get_pt(row, back_knee)
        b_ank = get_pt(row, back_ank)
        b_hip_angle = calc_angle(b_sh, b_hip, b_kn)
        b_kn_angle  = calc_angle(b_hip, b_kn, b_ank)
        knee_dominant_frames.append(
            (b_hip_angle > t["hip_straight_deg"]) and (b_kn_angle < t["knee_dominant_deg"])
        )
    results["knee_dominant_pct"] = round(float(np.mean(knee_dominant_frames)), 3) if knee_dominant_frames else None

    # 3. 골반 불안정: 로드 ~ 풋 플랜트
    hip_diffs = [abs(row["left_hip_y"] - row["right_hip_y"]) for _, row in load_to_fp.iterrows()]
    if hip_diffs:
        results["hip_instability"]  = round(float(np.std(hip_diffs)), 4)
        results["hip_unstable_pct"] = round(float(np.mean([d > t["hip_instability_std"] for d in hip_diffs])), 3)

    # 4. 몸통이 골반보다 앞섬: 컨택 시점
    if not contact_row.empty:
        row    = contact_row.iloc[0]
        sh_cx  = (row["left_shoulder_x"] + row["right_shoulder_x"]) / 2
        hip_cx = (row["left_hip_x"]      + row["right_hip_x"])      / 2
        results["trunk_ahead_at_contact"] = bool(sh_cx < hip_cx - t["trunk_ahead_threshold"])

    results["events"] = events
    return results


# ──────────────────────────────────────────
# 4. 키네마틱 시퀀스 분석
# ──────────────────────────────────────────

def analyze_kinematic_sequence(
    pose_df: pd.DataFrame,
    fps: float,
    events: dict,
    mode: str = "pitch",
) -> dict:
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)

    start_frame = events.get("foot_plant_frame", 0)
    end_frame   = events.get("release_frame" if mode == "pitch" else "contact_frame", len(df))

    seg_df = df[(df["frame"] >= start_frame) & (df["frame"] <= end_frame)].reset_index(drop=True)
    if len(seg_df) < 3:
        return {"error": "구간 내 프레임 부족"}

    dt = 1.0 / fps
    hip_angles, sh_angles, el_angles = [], [], []

    for _, row in seg_df.iterrows():
        hip_angles.append(rotation_angle_2d(get_pt(row, "left_hip"),      get_pt(row, "right_hip")))
        sh_angles.append( rotation_angle_2d(get_pt(row, "left_shoulder"), get_pt(row, "right_shoulder")))
        el_angles.append( calc_angle(get_pt(row, "right_shoulder"), get_pt(row, "right_elbow"), get_pt(row, "right_wrist")))

    hip_av = np.gradient(np.degrees(np.unwrap(np.radians(hip_angles))), dt)
    sh_av  = np.gradient(np.degrees(np.unwrap(np.radians(sh_angles))),  dt)
    el_av  = np.gradient(np.array(el_angles), dt)

    peak_vels, peak_times = {}, {}
    for seg, av in [("pelvis", hip_av), ("trunk", sh_av), ("elbow", el_av)]:
        abs_av = np.abs(av)
        idx    = int(np.argmax(abs_av))
        peak_vels[seg]  = round(float(abs_av[idx]), 1)
        peak_times[seg] = round(float(seg_df.loc[idx, "time_sec"]), 3)

    sequence     = sorted(peak_times.items(), key=lambda x: x[1])
    actual_order = [s[0] for s in sequence]
    ideal_order  = ["pelvis", "trunk", "elbow"]
    pv = peak_vels

    return {
        "actual_order":          actual_order,
        "ideal_order":           ideal_order,
        "is_proximal_to_distal": actual_order == ideal_order,
        "energy_build":          pv.get("trunk", 0) > pv.get("pelvis", 0) and pv.get("elbow", 0) > pv.get("trunk", 0),
        "peak_velocities":       peak_vels,
        "timing":                peak_times,
        "segment":               f"frame {start_frame} ~ {end_frame}",
    }


# ──────────────────────────────────────────
# 5. 전체 요약
# ──────────────────────────────────────────

def analyze_full(
    pose_df: pd.DataFrame,
    fps: float,
    hand: str = "R",
    mode: str = "pitch",
    thresholds: dict = None,  # ← classifier에서 받은 기준값
) -> dict:

    if mode == "pitch":
        events = detect_pitch_events(pose_df, fps, hand)
        ineff  = detect_pitching_inefficiency(pose_df, fps, hand, events, thresholds)
        ks     = analyze_kinematic_sequence(pose_df, fps, events, mode="pitch")
    else:
        events = detect_batting_events(pose_df, fps, hand)
        ineff  = detect_batting_inefficiency(pose_df, fps, hand, events)
        ks     = analyze_kinematic_sequence(pose_df, fps, events, mode="bat")

    summary = {
        "fp_sec":          events.get("foot_plant_sec"),
        "events_valid":    events.get("valid"),
        "ks_order":        str(ks.get("actual_order")),
        "ks_is_ptd":       ks.get("is_proximal_to_distal"),
        "ks_energy_build": ks.get("energy_build"),
        "ks_pelvis_v":     ks.get("peak_velocities", {}).get("pelvis"),
        "ks_trunk_v":      ks.get("peak_velocities", {}).get("trunk"),
        "ks_elbow_v":      ks.get("peak_velocities", {}).get("elbow"),
    }

    for k, v in ineff.items():
        if k != "events":
            summary[k] = v

    return summary


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    import os
    from classifier import classify_and_analyze

    CSV_PATH = "pose_output/hader_homerun_pose.csv"

    if not os.path.exists(CSV_PATH):
        print(f"파일 없음: {CSV_PATH}")
    else:
        pose_df = pd.read_csv(CSV_PATH)
        fps = 59.6

        # 1. 유형 자동 분류
        print("=" * 60)
        print("유형 분류")
        print("=" * 60)
        clf        = classify_and_analyze(pose_df, fps, mode="pitch")
        hand       = clf["hand"]
        thresholds = clf["thresholds"]

        # 2. 전체 분석 (thresholds 전달)
        print("\n" + "=" * 60)
        print("투구 분석 (구간 자동 감지)")
        print("=" * 60)
        summary = analyze_full(pose_df, fps, hand=hand, mode="pitch", thresholds=thresholds)

        print("\n[이벤트]")
        print(f"  풋 플랜트: {summary.get('fp_sec'):.2f}초")

        print("\n[키네마틱 시퀀스]")
        print(f"  실제 순서:      {summary.get('ks_order')}")
        print(f"  근위→원위 일치: {summary.get('ks_is_ptd')}")
        print(f"  에너지 증가:    {summary.get('ks_energy_build')}")
        print(f"  각속도: pelvis={summary.get('ks_pelvis_v')} trunk={summary.get('ks_trunk_v')} elbow={summary.get('ks_elbow_v')}")

        print("\n[비효율 동작 (구간별)]")
        keys = [
            "sway_detected", "sway_range",
            "early_open_pct", "early_open_skipped",
            "wrist_below_elbow_at_fp", "wrist_below_elbow_skipped",
            "trunk_ahead_at_fp",
            "knee_collapse_pct",
            "arm_flyout_pct",
            "late_release",
            "x_factor_max", "x_factor_status",
        ]
        for k in keys:
            if k in summary:
                print(f"  {k}: {summary[k]}")
