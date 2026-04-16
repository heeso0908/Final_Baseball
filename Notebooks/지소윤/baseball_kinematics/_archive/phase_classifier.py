"""
투구 단계 분류 LightGBM 학습 & 추론 모듈

학습 데이터: labeler.py로 만든 _train.csv 파일들
출력: 프레임별 투구 단계 예측 (0~4, -1=other)

투구 단계:
    -1: other (투구 외)
    0: windup
    1: stride
    2: cocking
    3: acceleration
    4: follow_through
"""

import numpy as np
import pandas as pd
import os
import json
import joblib
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


PHASE_LABELS = {
    -1: "other",
    0:  "windup",
    1:  "stride",
    2:  "cocking",
    3:  "acceleration",
    4:  "follow_through",
}

FEATURE_COLS = [
    "norm_r_hip_size", "norm_trunk_size",
    "norm_r_upper_arm", "norm_r_forearm",
    "norm_l_upper_arm", "norm_stride_len",
    "norm_knee_size", "norm_hip_dist",
    "r_hip_angle", "l_hip_angle",
    "r_knee_angle", "l_knee_angle",
    "r_elbow_angle", "l_elbow_angle",
    "r_shoulder_angle", "l_shoulder_angle",
    "r_wrist_y", "l_wrist_y",
    "r_wrist_x", "l_wrist_x",
    "r_elbow_y", "l_ankle_y", "r_ankle_y",
    "hip_center_x", "shoulder_center_x",
]

MODEL_PATH = "models/phase_classifier.pkl"


# ──────────────────────────────────────────
# 1. 학습 데이터 로드
# ──────────────────────────────────────────

def load_training_data(data_dir: str = "pose_output") -> pd.DataFrame:
    """
    pose_output 폴더의 모든 _train.csv 파일 로드 & 합치기
    """
    # OBP 자동 라벨링 파일 또는 수동 라벨링 파일 모두 지원
    files = [f for f in os.listdir(data_dir)
             if f.endswith("_train.csv") or f == "obp_train.csv"]
    if not files:
        raise FileNotFoundError(
            f"{data_dir}에 학습 데이터 없음. "
            "python obp_labeler.py 먼저 실행하세요."
        )

    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df = df[df["label"] != -1]  # other 제외
        dfs.append(df)
        print(f"  로드: {f} ({len(df)}행)")

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  → 총 {len(combined)}행 (other 제외)")
    return combined


# ──────────────────────────────────────────
# 2. LightGBM 학습
# ──────────────────────────────────────────

def train_phase_classifier(data_dir: str = "pose_output") -> lgb.LGBMClassifier:
    """
    투구 단계 분류 모델 학습

    Osawa et al. (2025) 방법론:
    - MediaPipe 33개 관절 좌표 → 정규화 피처
    - LightGBM 분류기
    - SHAP으로 피처 중요도 분석
    """
    print("[1] 학습 데이터 로드...")
    df = load_training_data(data_dir)

    X = df[FEATURE_COLS].fillna(0)
    y = df["label"]

    print(f"\n[2] 클래스 분포:")
    for label, count in y.value_counts().items():
        print(f"  {PHASE_LABELS[label]}: {count}프레임")

    # 학습/검증 분할
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n[3] LightGBM 학습 중...")
    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=8,
        class_weight="balanced",  # 클래스 불균형 처리
        random_state=42,
        verbose=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(50)],
    )

    # 평가
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"\n[4] 검증 정확도: {acc:.3f}")
    print(classification_report(y_val, y_pred, target_names=[PHASE_LABELS[i] for i in sorted(PHASE_LABELS.keys()) if i != -1]))

    # 피처 중요도
    importance_df = pd.DataFrame({
        "feature":    FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    print("\n[5] 피처 중요도 (상위 10개):")
    print(importance_df.head(10).to_string(index=False))

    # 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n✓ 모델 저장: {MODEL_PATH}")

    return model


# ──────────────────────────────────────────
# 3. 추론 (새 영상에 적용)
# ──────────────────────────────────────────

def extract_features(pose_df: pd.DataFrame) -> pd.DataFrame:
    """포즈 DataFrame → 피처 DataFrame"""
    rows = []
    for _, row in pose_df[pose_df["detected"]].iterrows():

        def pt(name):
            return np.array([row.get(f"{name}_x", 0), row.get(f"{name}_y", 0)])

        def dist(a, b):
            return float(np.linalg.norm(pt(a) - pt(b)))

        def angle(a, b, c):
            va = pt(a) - pt(b)
            vc = pt(c) - pt(b)
            cos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        trunk_len = dist("right_shoulder", "right_hip") + 1e-6

        rows.append({
            "frame":    int(row["frame"]),
            "time_sec": row["time_sec"],
            "norm_r_hip_size":    dist("right_hip", "right_knee") / trunk_len,
            "norm_trunk_size":    dist("right_shoulder", "right_hip") / trunk_len,
            "norm_r_upper_arm":   dist("right_shoulder", "right_elbow") / trunk_len,
            "norm_r_forearm":     dist("right_elbow", "right_wrist") / trunk_len,
            "norm_l_upper_arm":   dist("left_shoulder", "left_elbow") / trunk_len,
            "norm_stride_len":    dist("left_ankle", "right_ankle") / trunk_len,
            "norm_knee_size":     dist("right_knee", "right_ankle") / trunk_len,
            "norm_hip_dist":      dist("left_hip", "right_hip") / trunk_len,
            "r_hip_angle":        angle("right_shoulder", "right_hip", "right_knee"),
            "l_hip_angle":        angle("left_shoulder", "left_hip", "left_knee"),
            "r_knee_angle":       angle("right_hip", "right_knee", "right_ankle"),
            "l_knee_angle":       angle("left_hip", "left_knee", "left_ankle"),
            "r_elbow_angle":      angle("right_shoulder", "right_elbow", "right_wrist"),
            "l_elbow_angle":      angle("left_shoulder", "left_elbow", "left_wrist"),
            "r_shoulder_angle":   angle("right_elbow", "right_shoulder", "right_hip"),
            "l_shoulder_angle":   angle("left_elbow", "left_shoulder", "left_hip"),
            "r_wrist_y":          row.get("right_wrist_y", 0),
            "l_wrist_y":          row.get("left_wrist_y", 0),
            "r_wrist_x":          row.get("right_wrist_x", 0),
            "l_wrist_x":          row.get("left_wrist_x", 0),
            "r_elbow_y":          row.get("right_elbow_y", 0),
            "l_ankle_y":          row.get("left_ankle_y", 0),
            "r_ankle_y":          row.get("right_ankle_y", 0),
            "hip_center_x":       (row.get("left_hip_x", 0) + row.get("right_hip_x", 0)) / 2,
            "shoulder_center_x":  (row.get("left_shoulder_x", 0) + row.get("right_shoulder_x", 0)) / 2,
        })

    return pd.DataFrame(rows)


def predict_phases(
    pose_df: pd.DataFrame,
    model_path: str = MODEL_PATH,
) -> pd.DataFrame:
    """
    포즈 DataFrame → 프레임별 투구 단계 예측

    Returns:
        DataFrame with columns: frame, time_sec, phase, phase_name
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"모델 없음: {model_path}\ntrain_phase_classifier() 먼저 실행하세요.")

    model = joblib.load(model_path)
    feat_df = extract_features(pose_df)

    X = feat_df[FEATURE_COLS].fillna(0)
    preds = model.predict(X)

    feat_df["phase"]      = preds
    feat_df["phase_name"] = feat_df["phase"].map(PHASE_LABELS)

    return feat_df[["frame", "time_sec", "phase", "phase_name"]]


def get_phase_events(phase_df: pd.DataFrame, fps: float) -> dict:
    """
    단계 예측 결과 → 이벤트 타이밍 추출

    풋 플랜트: stride → cocking 전환 시점
    릴리즈:   acceleration → follow_through 전환 시점
    """
    events = {}

    # 각 단계 전환 시점 찾기
    prev_phase = None
    for _, row in phase_df.iterrows():
        curr_phase = row["phase"]

        if prev_phase == 1 and curr_phase == 2:  # stride → cocking = 풋 플랜트
            events["foot_plant_frame"] = int(row["frame"])
            events["foot_plant_sec"]   = float(row["time_sec"])

        if prev_phase == 3 and curr_phase == 4:  # acceleration → follow = 릴리즈
            events["release_frame"] = int(row["frame"])
            events["release_sec"]   = float(row["time_sec"])

        prev_phase = curr_phase

    # 투구 구간 (stride 시작 ~ follow_through 끝)
    pitch_frames = phase_df[phase_df["phase"].isin([1, 2, 3, 4])]
    if not pitch_frames.empty:
        events["pitch_start_frame"] = int(pitch_frames["frame"].min())
        events["pitch_end_frame"]   = int(pitch_frames["frame"].max())

    events["valid"] = "foot_plant_frame" in events and "release_frame" in events

    if events.get("valid"):
        print(f"  [LightGBM 이벤트] 풋 플랜트: {events['foot_plant_sec']:.2f}초 (frame {events['foot_plant_frame']}) | 릴리즈: {events['release_sec']:.2f}초 (frame {events['release_frame']})")
    else:
        print("  ⚠ 이벤트 감지 실패 → 규칙 기반으로 대체")

    return events


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        # 학습 모드
        print("=" * 60)
        print("투구 단계 분류 모델 학습")
        print("=" * 60)
        model = train_phase_classifier()

    else:
        # 추론 테스트
        CSV_PATH = "pose_output/hader_homerun_pose.csv"
        if not os.path.exists(CSV_PATH):
            print("포즈 CSV 없음")
        elif not os.path.exists(MODEL_PATH):
            print(f"모델 없음: {MODEL_PATH}")
            print("python phase_classifier.py train  ← 먼저 실행")
        else:
            pose_df = pd.read_csv(CSV_PATH)
            fps = 59.6

            print("투구 단계 예측 중...")
            phase_df = predict_phases(pose_df)
            print(phase_df["phase_name"].value_counts())

            events = get_phase_events(phase_df, fps)
            print(f"\n이벤트: {events}")
