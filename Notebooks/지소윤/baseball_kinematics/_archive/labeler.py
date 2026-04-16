"""
투구 단계 라벨링 도구
- 포즈 CSV를 불러와서 프레임별로 단계를 라벨링
- 라벨 저장 → LightGBM 학습 데이터로 사용

투구 5단계:
    0: windup      - 와인드업 (시작 ~ 다리 올리기)
    1: stride      - 스트라이드 (다리 올리기 ~ 앞발 착지 직전)
    2: cocking     - 암 코킹 (앞발 착지 ~ 최대 외회전)
    3: acceleration - 가속 (최대 외회전 ~ 릴리즈)
    4: follow_through - 팔로우스루 (릴리즈 이후)
    -1: other      - 투구 외 구간 (리플레이, 광고 등)

실행:
    streamlit run labeler.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json

PHASE_LABELS = {
    -1: "❌ other (투구 외)",
    0:  "🔵 windup",
    1:  "🟢 stride",
    2:  "🟡 cocking",
    3:  "🔴 acceleration",
    4:  "⚪ follow_through",
}

PHASE_COLORS = {
    -1: "#888888",
    0:  "#4488ff",
    1:  "#44bb44",
    2:  "#ffcc00",
    3:  "#ff4444",
    4:  "#aaaaaa",
}


def load_pose(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def save_labels(labels: dict, save_path: str):
    with open(save_path, "w") as f:
        json.dump(labels, f)
    st.success(f"저장 완료: {save_path}")


def load_labels(save_path: str) -> dict:
    if os.path.exists(save_path):
        with open(save_path) as f:
            return json.load(f)
    return {}


def build_training_data(pose_df: pd.DataFrame, labels: dict) -> pd.DataFrame:
    """
    포즈 DataFrame + 라벨 → LightGBM 학습용 DataFrame
    """
    rows = []
    for _, row in pose_df[pose_df["detected"]].iterrows():
        frame = str(int(row["frame"]))
        label = labels.get(frame, -1)

        # 피처 계산
        def pt(name):
            return np.array([row.get(f"{name}_x", 0), row.get(f"{name}_y", 0)])

        def dist(a, b):
            return float(np.linalg.norm(pt(a) - pt(b)))

        def angle(a, b, c):
            va = pt(a) - pt(b)
            vc = pt(c) - pt(b)
            cos = np.dot(va, vc) / (np.linalg.norm(va) * np.linalg.norm(vc) + 1e-6)
            return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))

        # 정규화 기준: 몸통 길이 (어깨 ~ 힙)
        trunk_len = dist("right_shoulder", "right_hip") + 1e-6

        features = {
            "frame":    int(row["frame"]),
            "time_sec": row["time_sec"],

            # 정규화된 거리
            "norm_r_hip_size":    dist("right_hip", "right_knee") / trunk_len,
            "norm_trunk_size":    dist("right_shoulder", "right_hip") / trunk_len,
            "norm_r_upper_arm":   dist("right_shoulder", "right_elbow") / trunk_len,
            "norm_r_forearm":     dist("right_elbow", "right_wrist") / trunk_len,
            "norm_l_upper_arm":   dist("left_shoulder", "left_elbow") / trunk_len,
            "norm_stride_len":    dist("left_ankle", "right_ankle") / trunk_len,
            "norm_knee_size":     dist("right_knee", "right_ankle") / trunk_len,
            "norm_hip_dist":      dist("left_hip", "right_hip") / trunk_len,

            # 관절 각도
            "r_hip_angle":        angle("right_shoulder", "right_hip", "right_knee"),
            "l_hip_angle":        angle("left_shoulder", "left_hip", "left_knee"),
            "r_knee_angle":       angle("right_hip", "right_knee", "right_ankle"),
            "l_knee_angle":       angle("left_hip", "left_knee", "left_ankle"),
            "r_elbow_angle":      angle("right_shoulder", "right_elbow", "right_wrist"),
            "l_elbow_angle":      angle("left_shoulder", "left_elbow", "left_wrist"),
            "r_shoulder_angle":   angle("right_elbow", "right_shoulder", "right_hip"),
            "l_shoulder_angle":   angle("left_elbow", "left_shoulder", "left_hip"),

            # 위치 피처
            "r_wrist_y":          row.get("right_wrist_y", 0),
            "l_wrist_y":          row.get("left_wrist_y", 0),
            "r_wrist_x":          row.get("right_wrist_x", 0),
            "l_wrist_x":          row.get("left_wrist_x", 0),
            "r_elbow_y":          row.get("right_elbow_y", 0),
            "l_ankle_y":          row.get("left_ankle_y", 0),
            "r_ankle_y":          row.get("right_ankle_y", 0),
            "hip_center_x":       (row.get("left_hip_x", 0) + row.get("right_hip_x", 0)) / 2,
            "shoulder_center_x":  (row.get("left_shoulder_x", 0) + row.get("right_shoulder_x", 0)) / 2,

            "label": label,
        }
        rows.append(features)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────

st.set_page_config(page_title="투구 단계 라벨러", layout="wide")
st.title("⚾ 투구 단계 라벨링 도구")

# 사이드바: 파일 선택
st.sidebar.header("파일 설정")
pose_dir = st.sidebar.text_input("포즈 CSV 폴더", value="pose_output")
csv_files = [f for f in os.listdir(pose_dir) if f.endswith("_pose.csv")] if os.path.exists(pose_dir) else []

if not csv_files:
    st.warning("pose_output 폴더에 _pose.csv 파일이 없어요. pose_extractor.py를 먼저 실행해주세요.")
    st.stop()

selected_file = st.sidebar.selectbox("포즈 CSV 선택", csv_files)
csv_path   = os.path.join(pose_dir, selected_file)
label_path = csv_path.replace("_pose.csv", "_labels.json")
train_path = csv_path.replace("_pose.csv", "_train.csv")

pose_df = load_pose(csv_path)
labels  = load_labels(label_path)

st.sidebar.markdown(f"**총 프레임**: {len(pose_df)}")
st.sidebar.markdown(f"**감지된 프레임**: {pose_df['detected'].sum()}")
st.sidebar.markdown(f"**라벨링된 프레임**: {len(labels)}")

# ── 구간 라벨링 (범위 지정 방식) ──
st.header("구간 라벨링")
st.info("시작 프레임 ~ 끝 프레임 범위를 지정하고 단계를 선택하면 해당 구간 전체에 라벨이 적용돼요.")

col1, col2, col3, col4 = st.columns([2, 2, 3, 2])

max_frame = int(pose_df["frame"].max())

with col1:
    start_frame = st.number_input("시작 프레임", min_value=0, max_value=max_frame, value=0)
with col2:
    end_frame = st.number_input("끝 프레임", min_value=0, max_value=max_frame, value=min(30, max_frame))
with col3:
    phase = st.selectbox("단계", options=list(PHASE_LABELS.keys()), format_func=lambda x: PHASE_LABELS[x])
with col4:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✅ 적용", use_container_width=True):
        for f in range(int(start_frame), int(end_frame) + 1):
            labels[str(f)] = phase
        save_labels(labels, label_path)
        st.rerun()

# ── 현재 라벨 현황 시각화 ──
st.header("라벨 현황")

frames = list(range(max_frame + 1))
label_vals = [labels.get(str(f), -1) for f in frames]

# 라벨 분포 바 차트
label_df = pd.DataFrame({"frame": frames, "label": label_vals})
label_counts = label_df["label"].value_counts().rename(index=PHASE_LABELS)
st.bar_chart(label_counts)

# 타임라인 시각화
st.subheader("타임라인")
timeline_data = []
for f, l in zip(frames, label_vals):
    if l != -1:
        timeline_data.append({"frame": f, "phase": PHASE_LABELS[l], "value": 1})

if timeline_data:
    tl_df = pd.DataFrame(timeline_data)
    st.dataframe(
        tl_df.groupby("phase")["frame"].agg(["min", "max", "count"]).rename(
            columns={"min": "시작 프레임", "max": "끝 프레임", "count": "프레임 수"}
        ),
        use_container_width=True
    )

# ── 학습 데이터 생성 ──
st.header("학습 데이터 생성")

labeled_count = sum(1 for v in labels.values() if v != -1)
st.markdown(f"라벨링된 프레임: **{labeled_count}개** / 전체 감지 프레임: **{int(pose_df['detected'].sum())}개**")

if labeled_count < 50:
    st.warning("최소 50개 프레임 이상 라벨링 후 학습 데이터를 생성하세요.")
else:
    if st.button("📊 학습 데이터 생성 & 저장", use_container_width=True):
        train_df = build_training_data(pose_df, labels)
        labeled_train = train_df[train_df["label"] != -1]
        labeled_train.to_csv(train_path, index=False)
        st.success(f"저장 완료: {train_path} ({len(labeled_train)}행)")
        st.dataframe(labeled_train["label"].value_counts().rename(index=PHASE_LABELS))

# ── 단계별 라벨 초기화 ──
st.sidebar.markdown("---")
st.sidebar.subheader("⚠️ 초기화")
if st.sidebar.button("전체 라벨 초기화"):
    save_labels({}, label_path)
    st.rerun()
