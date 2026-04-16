"""
OBP 데이터 기반 자동 라벨링
- joint_angles.csv (시계열 관절 각도) + 타이밍 컬럼으로 5단계 라벨 자동 생성
- MediaPipe 피처 형식으로 변환 → _train.csv 저장
- phase_classifier.py train으로 바로 학습 가능

실행:
    python obp_labeler.py
"""

import pandas as pd
import numpy as np
import os

OBP_DIR      = "openbiomechanics/baseball_pitching/data"
JOINT_ANGLES = os.path.join(OBP_DIR, "full_sig/joint_angles/joint_angles.csv")
POI_CSV      = os.path.join(OBP_DIR, "poi/poi_metrics.csv")
OUTPUT_PATH  = "pose_output/obp_train.csv"

PHASE_LABELS = {
    "windup":        0,
    "stride":        1,
    "cocking":       2,
    "acceleration":  3,
    "follow_through":4,
}


# ──────────────────────────────────────────
# 1. 타이밍 기반 라벨 생성
# ──────────────────────────────────────────

def assign_phase_label(t: float, pkh: float, fp100: float,
                        mer: float, br: float, mir: float) -> int:
    """
    프레임 시각(t)과 이벤트 타이밍으로 투구 단계 라벨 반환

    windup       : 0 ~ pkh
    stride       : pkh ~ fp100
    cocking      : fp100 ~ mer
    acceleration : mer ~ br
    follow_through: br ~ mir
    """
    if t < pkh:
        return 0  # windup
    elif t < fp100:
        return 1  # stride
    elif t < mer:
        return 2  # cocking
    elif t < br:
        return 3  # acceleration
    elif t < mir:
        return 4  # follow_through
    else:
        return -1  # other (MIR 이후)


# ──────────────────────────────────────────
# 2. OBP 관절 각도 → MediaPipe 피처 형식 변환
# ──────────────────────────────────────────

def obp_to_mediapipe_features(row: pd.Series, trunk_len: float = 1.0) -> dict:
    """
    OBP joint_angles 컬럼 → phase_classifier.py FEATURE_COLS 형식으로 변환

    OBP는 3D 관절 각도(x,y,z)를 제공
    MediaPipe는 2D 투영 기반 각도를 사용

    매핑 전략:
    - OBP의 x축 각도 (시상면) = 사이드뷰에서 보이는 각도와 가장 유사
    - 정규화된 거리는 OBP 각도 기반으로 근사

    주의: 완벽한 변환은 불가능하지만 학습 데이터로 충분히 유효
    """

    def safe(col, default=0.0):
        return float(row.get(col, default)) if pd.notna(row.get(col, default)) else default

    # OBP 관절 각도 (x축 = 시상면 굴곡/신전)
    r_hip_x     = safe("rear_hip_angle_x")
    r_knee_x    = safe("rear_knee_angle_x")
    l_hip_x     = safe("lead_hip_angle_x")
    l_knee_x    = safe("lead_knee_angle_x")
    elbow_x     = safe("elbow_angle_x")
    shoulder_x  = safe("shoulder_angle_x")
    torso_z     = safe("torso_angle_z")    # 몸통 회전
    pelvis_z    = safe("pelvis_angle_z")   # 골반 회전
    torso_pel_z = safe("torso_pelvis_angle_z")  # X-Factor

    # 정규화된 거리 (각도에서 근사)
    # 관절 각도가 작을수록 가까움 (굴곡 = 짧아 보임)
    def angle_to_norm_dist(angle_deg, base=1.0):
        """관절 각도 → 정규화된 분절 길이 근사"""
        return base * abs(np.cos(np.radians(angle_deg / 2)))

    features = {
        # 정규화 거리 (근사값)
        "norm_r_hip_size":   angle_to_norm_dist(r_knee_x),
        "norm_trunk_size":   1.0,  # 기준값
        "norm_r_upper_arm":  angle_to_norm_dist(elbow_x),
        "norm_r_forearm":    angle_to_norm_dist(elbow_x * 0.7),
        "norm_l_upper_arm":  angle_to_norm_dist(safe("glove_elbow_angle_x")),
        "norm_stride_len":   abs(np.sin(np.radians(l_hip_x))) * 1.5,
        "norm_knee_size":    angle_to_norm_dist(r_knee_x),
        "norm_hip_dist":     abs(np.sin(np.radians(pelvis_z))) * 0.5 + 0.3,

        # 관절 각도 (OBP x축 = 사이드뷰 굴곡각과 유사)
        "r_hip_angle":       180 - abs(r_hip_x),
        "l_hip_angle":       180 - abs(l_hip_x),
        "r_knee_angle":      180 - abs(r_knee_x),
        "l_knee_angle":      180 - abs(l_knee_x),
        "r_elbow_angle":     180 - abs(elbow_x),
        "l_elbow_angle":     180 - abs(safe("glove_elbow_angle_x")),
        "r_shoulder_angle":  abs(shoulder_x),
        "l_shoulder_angle":  abs(safe("glove_shoulder_angle_x")),

        # 위치 피처 (각도에서 간접 추정)
        "r_wrist_y":         0.5 - np.sin(np.radians(shoulder_x)) * 0.3,
        "l_wrist_y":         0.5 - np.sin(np.radians(safe("glove_shoulder_angle_x"))) * 0.2,
        "r_wrist_x":         0.5 + np.cos(np.radians(torso_z)) * 0.2,
        "l_wrist_x":         0.5 - np.cos(np.radians(torso_z)) * 0.15,
        "r_elbow_y":         0.5 - np.sin(np.radians(shoulder_x)) * 0.15,
        "l_ankle_y":         0.8 + np.sin(np.radians(l_knee_x)) * 0.1,
        "r_ankle_y":         0.8 + np.sin(np.radians(r_knee_x)) * 0.1,
        "hip_center_x":      0.45 + np.sin(np.radians(pelvis_z)) * 0.05,
        "shoulder_center_x": 0.45 + np.sin(np.radians(torso_z)) * 0.08,
    }

    return features


# ──────────────────────────────────────────
# 3. 전체 파이프라인
# ──────────────────────────────────────────

def generate_training_data(
    max_pitches: int = None,
    min_frames_per_phase: int = 5,
) -> pd.DataFrame:

    print("OBP 데이터 로드 중...")
    joint_df = pd.read_csv(JOINT_ANGLES)
    print(f"  joint_angles: {len(joint_df)}행 ({joint_df['session_pitch'].nunique()}개 투구)")

    # 타이밍 컬럼이 joint_angles에 포함돼 있음
    merged = joint_df.dropna(subset=["pkh_time", "fp_100_time", "MER_time", "BR_time", "MIR_time"])
    print(f"  타이밍 완전한 행: {len(merged)}행")

    if max_pitches:
        pitches = merged["session_pitch"].unique()[:max_pitches]
        merged  = merged[merged["session_pitch"].isin(pitches)]
        print(f"  샘플링: {max_pitches}개 투구")

    # 라벨 생성
    print("\n라벨 생성 중...")
    rows = []
    pitch_count = 0

    for pitch_id, group in merged.groupby("session_pitch"):
        group = group.sort_values("time").reset_index(drop=True)

        pkh   = group["pkh_time"].iloc[0]
        fp100 = group["fp_100_time"].iloc[0]
        mer   = group["MER_time"].iloc[0]
        br    = group["BR_time"].iloc[0]
        mir   = group["MIR_time"].iloc[0]

        pitch_rows = []
        for _, row in group.iterrows():
            t     = row["time"]
            label = assign_phase_label(t, pkh, fp100, mer, br, mir)

            if label == -1:
                continue

            features              = obp_to_mediapipe_features(row)
            features["label"]         = label
            features["session_pitch"] = pitch_id
            features["time"]          = t
            pitch_rows.append(features)

        if pitch_rows:
            rows.extend(pitch_rows)
            pitch_count += 1

    df = pd.DataFrame(rows)

    print(f"\n[결과]")
    print(f"  총 {pitch_count}개 투구, {len(df)}프레임")
    print(f"\n[단계별 분포]")
    phase_names = {0:"windup", 1:"stride", 2:"cocking", 3:"acceleration", 4:"follow_through"}
    for label, count in df["label"].value_counts().sort_index().items():
        print(f"  {phase_names[label]}: {count}프레임")

    return df

# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("OBP 데이터 → LightGBM 학습 데이터 생성")
    print("=" * 60)

    df = generate_training_data(max_pitches=None)

    os.makedirs("pose_output", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✓ 저장 완료: {OUTPUT_PATH} ({len(df)}행)")
    print(f"\n다음 단계:")
    print(f"  python phase_classifier.py train")
