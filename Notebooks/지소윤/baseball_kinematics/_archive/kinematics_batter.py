"""
타자 키네마틱 분석 모듈
- 센터필드 뷰 영상 기반 (하체/몸통 중심)
- 골반 회전, 어깨-골반 분리각, 체중이동, 무릎 굽힘 분석

실행:
    python kinematics_batter.py
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────
# 기본 유틸
# ──────────────────────────────────────────

def pt(row, name):
    return np.array([row.get(f"{name}_x", 0), row.get(f"{name}_y", 0)])

def dist(row, a, b):
    return float(np.linalg.norm(pt(row, a) - pt(row, b)))

def angle(row, a, b, c):
    va = pt(row, a) - pt(row, b)
    vc = pt(row, c) - pt(row, b)
    cos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

def smooth(series, window=5):
    return series.rolling(window, center=True, min_periods=1).mean()


# ──────────────────────────────────────────
# 1. 스윙 이벤트 감지
# ──────────────────────────────────────────

def detect_swing_events(df: pd.DataFrame, fps: float) -> dict:
    """
    스윙 이벤트 타이밍 감지
    - 풋플랜트: 앞발 ankle y좌표 안정화 시점
    - 컨택: 골반 회전 속도 최대 시점
    """
    detected = df[df["detected"]].copy().reset_index(drop=True)
    events = {}

    # 골반 중심 x좌표
    detected["hip_center_x"] = (
        detected["left_hip_x"].fillna(0) + detected["right_hip_x"].fillna(0)
    ) / 2

    # 어깨 중심 x좌표
    detected["shoulder_center_x"] = (
        detected["left_shoulder_x"].fillna(0) + detected["right_shoulder_x"].fillna(0)
    ) / 2

    # 골반 회전 속도 (x좌표 변화율)
    detected["hip_rot_speed"] = detected["hip_center_x"].diff().abs() * fps
    detected["hip_rot_speed"] = smooth(detected["hip_rot_speed"])

    # 컨택 = 골반 회전 속도 최대 시점
    if not detected["hip_rot_speed"].isna().all():
        contact_idx = detected["hip_rot_speed"].idxmax()
        events["contact_frame"] = int(detected.loc[contact_idx, "frame"])
        events["contact_sec"]   = float(detected.loc[contact_idx, "time_sec"])

    # 풋플랜트 = 앞발(우타자=left_ankle) y좌표 안정화
    # 우타자: left_ankle이 앞발
    detected["lead_ankle_y"] = detected["left_ankle_y"]
    detected["ankle_diff"]   = detected["lead_ankle_y"].diff().abs()
    detected["ankle_diff"]   = smooth(detected["ankle_diff"])

    # 발이 안정된 시점 (ankle 움직임 급격히 줄어드는 지점)
    threshold = detected["ankle_diff"].quantile(0.3)
    stable    = detected[detected["ankle_diff"] < threshold]
    if not stable.empty:
        fp_idx = stable.index[0]
        events["foot_plant_frame"] = int(detected.loc[fp_idx, "frame"])
        events["foot_plant_sec"]   = float(detected.loc[fp_idx, "time_sec"])

    events["valid"] = "contact_frame" in events

    if events.get("valid"):
        print(f"  [이벤트] 풋플랜트: {events.get('foot_plant_sec', 'N/A'):.2f}초 | 컨택: {events['contact_sec']:.2f}초")

    return events


# ──────────────────────────────────────────
# 2. 키네마틱 분석
# ──────────────────────────────────────────

def analyze_batter(df: pd.DataFrame, fps: float, batter_side: str = "right") -> dict:
    """
    타자 키네마틱 분석

    Parameters:
        df: 포즈 CSV DataFrame
        fps: 영상 FPS
        batter_side: "right" (우타) or "left" (좌타)

    Returns:
        분석 결과 dict
    """
    detected = df[df["detected"]].copy().reset_index(drop=True)
    if len(detected) < 10:
        return {"error": "감지된 프레임 부족"}

    summary = {}

    # 이벤트 감지
    events = detect_swing_events(detected, fps)
    summary.update(events)

    # 우타/좌타에 따라 앞발/뒷발 설정
    if batter_side == "right":
        lead_hip    = "left_hip"
        trail_hip   = "right_hip"
        lead_knee   = "left_knee"
        trail_knee  = "right_knee"
        lead_ankle  = "left_ankle"
        lead_shoulder  = "left_shoulder"
        trail_shoulder = "right_shoulder"
    else:
        lead_hip    = "right_hip"
        trail_hip   = "left_hip"
        lead_knee   = "right_knee"
        trail_knee  = "left_knee"
        lead_ankle  = "right_ankle"
        lead_shoulder  = "right_shoulder"
        trail_shoulder = "left_shoulder"

    # ── 골반 회전 ──
    detected["hip_width"] = detected.apply(
        lambda r: dist(r, "left_hip", "right_hip"), axis=1
    )
    detected["hip_center_x"] = (
        detected["left_hip_x"].fillna(0) + detected["right_hip_x"].fillna(0)
    ) / 2
    detected["hip_center_x"] = smooth(detected["hip_center_x"])
    detected["hip_rot_speed"] = detected["hip_center_x"].diff().abs() * fps

    summary["hip_rot_speed_max"] = round(float(detected["hip_rot_speed"].max()), 2)
    summary["hip_rot_speed_mean"] = round(float(detected["hip_rot_speed"].mean()), 2)

    # ── 어깨 회전 ──
    detected["shoulder_center_x"] = (
        detected["left_shoulder_x"].fillna(0) + detected["right_shoulder_x"].fillna(0)
    ) / 2
    detected["shoulder_center_x"] = smooth(detected["shoulder_center_x"])
    detected["shoulder_rot_speed"] = detected["shoulder_center_x"].diff().abs() * fps

    summary["shoulder_rot_speed_max"] = round(float(detected["shoulder_rot_speed"].max()), 2)

    # ── X-Factor (어깨-골반 분리각) ──
    detected["x_factor"] = (
        detected["shoulder_center_x"] - detected["hip_center_x"]
    ).abs() * 100  # 픽셀 비율 → 스케일

    summary["x_factor_max"]  = round(float(detected["x_factor"].max()), 2)
    summary["x_factor_mean"] = round(float(detected["x_factor"].mean()), 2)

    # ── 무릎 굽힘각 ──
    detected["lead_knee_angle"] = detected.apply(
        lambda r: angle(r, lead_hip, lead_knee, lead_ankle), axis=1
    )
    detected["trail_knee_angle"] = detected.apply(
        lambda r: angle(r, trail_hip, trail_knee, lead_ankle), axis=1
    )

    summary["lead_knee_angle_mean"]  = round(float(detected["lead_knee_angle"].mean()), 1)
    summary["trail_knee_angle_mean"] = round(float(detected["trail_knee_angle"].mean()), 1)
    summary["lead_knee_angle_min"]   = round(float(detected["lead_knee_angle"].min()), 1)

    # ── 체중이동 ──
    hip_x_range = detected["hip_center_x"].max() - detected["hip_center_x"].min()
    summary["weight_transfer"] = round(float(hip_x_range * 100), 2)  # % 단위

    # ── 키네마틱 시퀀스 ──
    # 골반 → 어깨 순서 확인
    hip_peak_idx      = detected["hip_rot_speed"].idxmax()
    shoulder_peak_idx = detected["shoulder_rot_speed"].idxmax()

    hip_peak_frame      = detected.loc[hip_peak_idx, "frame"] if hip_peak_idx in detected.index else 0
    shoulder_peak_frame = detected.loc[shoulder_peak_idx, "frame"] if shoulder_peak_idx in detected.index else 0

    summary["ks_hip_before_shoulder"] = bool(hip_peak_frame < shoulder_peak_frame)
    summary["ks_hip_peak_frame"]      = int(hip_peak_frame)
    summary["ks_shoulder_peak_frame"] = int(shoulder_peak_frame)

    # ── 비효율 동작 감지 ──

    # 조기 체중이동 (풋플랜트 전에 hip이 너무 앞으로)
    if "foot_plant_frame" in events:
        fp_frame = events["foot_plant_frame"]
        before_fp = detected[detected["frame"] < fp_frame]
        if not before_fp.empty:
            hip_move_before = before_fp["hip_center_x"].std()
            summary["early_weight_transfer"] = bool(hip_move_before > 0.02)
        else:
            summary["early_weight_transfer"] = False
    else:
        summary["early_weight_transfer"] = False

    # 무릎 무너짐 (lead knee angle < 150도)
    knee_collapse_frames = (detected["lead_knee_angle"] < 150).sum()
    summary["knee_collapse_pct"] = round(float(knee_collapse_frames / len(detected)), 3)

    # 상체 선행 (shoulder가 hip보다 먼저 회전)
    summary["trunk_ahead"] = not summary["ks_hip_before_shoulder"]

    return summary


# ──────────────────────────────────────────
# 3. 결과 출력
# ──────────────────────────────────────────

def print_batter_report(summary: dict, player_name: str = "타자"):
    print(f"\n{'='*60}")
    print(f"타자 키네마틱 분석: {player_name}")
    print(f"{'='*60}")

    print("\n[이벤트]")
    print(f"  풋플랜트: {summary.get('foot_plant_sec', 'N/A')}초")
    print(f"  컨택:     {summary.get('contact_sec', 'N/A')}초")

    print("\n[회전 속도]")
    print(f"  골반 최대 회전속도:  {summary.get('hip_rot_speed_max', 'N/A')}")
    print(f"  어깨 최대 회전속도:  {summary.get('shoulder_rot_speed_max', 'N/A')}")

    print("\n[X-Factor (어깨-골반 분리)]")
    print(f"  최대: {summary.get('x_factor_max', 'N/A')}")
    print(f"  평균: {summary.get('x_factor_mean', 'N/A')}")

    print("\n[하체]")
    print(f"  앞발 무릎각 평균: {summary.get('lead_knee_angle_mean', 'N/A')}°")
    print(f"  앞발 무릎각 최소: {summary.get('lead_knee_angle_min', 'N/A')}°")
    print(f"  체중이동 범위:    {summary.get('weight_transfer', 'N/A')}")

    print("\n[키네마틱 시퀀스]")
    ks = summary.get('ks_hip_before_shoulder', False)
    print(f"  골반 → 어깨 순서: {'✅ 정상' if ks else '❌ 상체 선행'}")

    print("\n[비효율 동작]")
    print(f"  조기 체중이동: {'있음' if summary.get('early_weight_transfer') else '없음'}")
    print(f"  무릎 무너짐:   {summary.get('knee_collapse_pct', 0)*100:.1f}% 프레임")
    print(f"  상체 선행:     {'있음' if summary.get('trunk_ahead') else '없음'}")


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    CSV_PATH    = "pose_output/judge_homerun_yolo_pose.csv"
    PLAYER_NAME = "Aaron Judge"
    FPS         = 59.85
    SWING_END   = 4.0  # 스윙 구간 (초)

    df    = pd.read_csv(CSV_PATH)
    swing = df[df["time_sec"] <= SWING_END].copy()

    print(f"스윙 구간: {len(swing)}프레임 | 감지율: {swing['detected'].mean()*100:.1f}%")

    summary = analyze_batter(swing, FPS, batter_side="right")
    print_batter_report(summary, PLAYER_NAME)
