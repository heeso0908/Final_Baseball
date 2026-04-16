"""
영상 구간 자동 추출 모듈
- 포즈 감지율이 높은 구간만 사용
- 실제 투구/타격 구간 자동 감지

방법:
1. 전체 영상에서 포즈 감지율 계산
2. 감지율이 낮은 구간 (광고, 리플레이 등) 제거
3. 연속된 감지 구간 중 가장 긴 구간 선택
4. 해당 구간만 잘라서 분석에 사용
"""

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")


LANDMARKS = {
    "nose": 0,
    "left_shoulder": 11,  "right_shoulder": 12,
    "left_elbow": 13,     "right_elbow": 14,
    "left_wrist": 15,     "right_wrist": 16,
    "left_hip": 23,       "right_hip": 24,
    "left_knee": 25,      "right_knee": 26,
    "left_ankle": 27,     "right_ankle": 28,
}


# ──────────────────────────────────────────
# 1. 포즈 감지율 기반 유효 구간 탐지
# ──────────────────────────────────────────

def find_valid_segment(
    pose_df: pd.DataFrame,
    min_detection_rate: float = 0.7,  # 구간 내 최소 감지율
    min_segment_frames: int = 30,      # 최소 유효 프레임 수
    window_size: int = 15,             # 슬라이딩 윈도우 크기
) -> tuple[int, int]:
    """
    포즈 감지율이 높은 가장 긴 연속 구간 탐지

    Returns:
        (start_frame, end_frame) - 유효 구간
    """
    detected = pose_df["detected"].astype(int).values
    n = len(detected)

    # 슬라이딩 윈도우로 구간별 감지율 계산
    window_rates = []
    for i in range(n - window_size + 1):
        rate = detected[i:i + window_size].mean()
        window_rates.append(rate)
    window_rates = np.array(window_rates)

    # 감지율이 기준 이상인 프레임 마스크
    valid_mask = np.zeros(n, dtype=bool)
    for i, rate in enumerate(window_rates):
        if rate >= min_detection_rate:
            valid_mask[i:i + window_size] = True

    # 연속된 유효 구간 찾기
    segments = []
    in_segment = False
    seg_start = 0

    for i, v in enumerate(valid_mask):
        if v and not in_segment:
            seg_start = i
            in_segment = True
        elif not v and in_segment:
            seg_end = i - 1
            if seg_end - seg_start >= min_segment_frames:
                segments.append((seg_start, seg_end))
            in_segment = False

    if in_segment:
        seg_end = n - 1
        if seg_end - seg_start >= min_segment_frames:
            segments.append((seg_start, seg_end))

    if not segments:
        print(f"  ⚠ 유효 구간 없음 → 전체 구간 사용")
        return 0, n - 1

    # 가장 긴 구간 선택
    best = max(segments, key=lambda s: s[1] - s[0])
    print(f"  유효 구간: frame {best[0]} ~ {best[1]} ({best[1]-best[0]+1}프레임, {len(segments)}개 구간 중 선택)")

    return best[0], best[1]


# ──────────────────────────────────────────
# 2. 투구 구간 정밀 탐지
# ──────────────────────────────────────────

def find_pitch_segment(
    pose_df: pd.DataFrame,
    fps: float,
    pitcher_hand: str = "R",
    valid_start: int = 0,
    valid_end: int = None,
) -> tuple[int, int]:
    """
    유효 구간 내에서 실제 투구 동작 구간 탐지

    방법:
    - 투구팔 손목 x좌표의 급격한 변화 구간 탐지
    - 와인드업 시작 ~ 릴리즈 후 팔로우스루 끝까지

    Returns:
        (pitch_start_frame, pitch_end_frame)
    """
    df = pose_df[pose_df["detected"]].copy().reset_index(drop=True)

    if valid_end is None:
        valid_end = int(df["frame"].max())

    # 유효 구간만 사용
    seg = df[(df["frame"] >= valid_start) & (df["frame"] <= valid_end)].copy()
    seg = seg.reset_index(drop=True)

    if len(seg) < 10:
        return valid_start, valid_end

    throw_wrist = "right_wrist" if pitcher_hand == "R" else "left_wrist"
    front_ankle = "left_ankle"  if pitcher_hand == "R" else "right_ankle"

    # 손목 x 속도 (앞으로 이동 속도)
    wrist_x = seg[f"{throw_wrist}_x"].interpolate().values
    wrist_vx = np.abs(np.gradient(wrist_x, 1.0 / fps))

    # 앞발 발목 y (풋 플랜트 감지)
    ankle_y = seg[f"{front_ankle}_y"].interpolate().values

    # 투구 시작: 손목 속도가 처음으로 임계값 넘는 시점 (와인드업)
    vx_threshold = np.percentile(wrist_vx, 60)  # 상위 40% 활동 구간
    active_frames = np.where(wrist_vx > vx_threshold)[0]

    if len(active_frames) == 0:
        return valid_start, valid_end

    # 활동 구간의 시작과 끝
    pitch_start_local = max(0, int(active_frames[0]) - int(fps * 0.5))  # 0.5초 여유
    pitch_end_local   = min(len(seg) - 1, int(active_frames[-1]) + int(fps * 0.3))

    pitch_start_frame = int(seg.loc[pitch_start_local, "frame"]) if pitch_start_local < len(seg) else valid_start
    pitch_end_frame   = int(seg.loc[pitch_end_local,   "frame"]) if pitch_end_local   < len(seg) else valid_end

    duration = (pitch_end_frame - pitch_start_frame) / fps
    print(f"  투구 구간: frame {pitch_start_frame} ~ {pitch_end_frame} ({duration:.2f}초)")

    return pitch_start_frame, pitch_end_frame


# ──────────────────────────────────────────
# 3. 포즈 DataFrame 구간 자르기
# ──────────────────────────────────────────

def slice_pose_df(
    pose_df: pd.DataFrame,
    fps: float,
    mode: str = "pitch",
    pitcher_hand: str = "R",
    min_detection_rate: float = 0.7,
) -> pd.DataFrame:
    """
    포즈 DataFrame에서 유효한 투구/타격 구간만 추출

    Returns:
        슬라이싱된 pose_df (frame 번호 유지)
    """
    total_frames = len(pose_df)
    total_rate   = pose_df["detected"].mean()
    print(f"  전체 감지율: {total_rate:.1%} ({total_frames}프레임)")

    # 이미 감지율이 충분하면 그대로 사용
    if total_rate >= min_detection_rate:
        print(f"  감지율 충분 → 전체 구간 사용")
        return pose_df

    # 유효 구간 탐지
    valid_start, valid_end = find_valid_segment(
        pose_df,
        min_detection_rate=min_detection_rate,
    )

    # 투구 구간 정밀 탐지
    if mode == "pitch":
        pitch_start, pitch_end = find_pitch_segment(
            pose_df, fps, pitcher_hand,
            valid_start=valid_start,
            valid_end=valid_end,
        )
        sliced = pose_df[
            (pose_df["frame"] >= pitch_start) &
            (pose_df["frame"] <= pitch_end)
        ].copy()
    else:
        sliced = pose_df[
            (pose_df["frame"] >= valid_start) &
            (pose_df["frame"] <= valid_end)
        ].copy()

    sliced_rate = sliced["detected"].mean()
    print(f"  슬라이싱 후 감지율: {sliced_rate:.1%} ({len(sliced)}프레임)")

    return sliced


# ──────────────────────────────────────────
# 4. 유형 분류 신뢰도 체크
# ──────────────────────────────────────────

def check_classification_reliability(
    pose_df: pd.DataFrame,
    clf_result: dict,
    min_detection_rate: float = 0.8,
) -> dict:
    """
    유형 분류 결과의 신뢰도 체크

    감지율이 낮으면 분류 결과가 부정확할 수 있음
    """
    detection_rate = pose_df["detected"].mean()
    hand_confidence = clf_result["type"].get("hand_confidence", 0)

    reliable = (
        detection_rate >= min_detection_rate and
        hand_confidence >= 0.6
    )

    clf_result["type"]["detection_rate"]    = round(float(detection_rate), 3)
    clf_result["type"]["overall_reliable"]  = reliable

    if not reliable:
        print(f"  ⚠ 분류 신뢰도 낮음 (감지율: {detection_rate:.1%}, 손 신뢰도: {hand_confidence:.0%})")
        print(f"    → 결과 해석 시 주의 필요")

    return clf_result


# ──────────────────────────────────────────
# 테스트
# ──────────────────────────────────────────

if __name__ == "__main__":
    import os

    # 감지율 낮은 영상으로 테스트 (홈런 영상)
    CSV_PATH = "pose_output/2fde4f59-a085-4b7d-845a-d88524fc07a4_pose.csv"

    if not os.path.exists(CSV_PATH):
        # 기존 pose csv 사용
        CSV_PATH = "pose_output/hader_homerun_pose.csv"

    if not os.path.exists(CSV_PATH):
        print("포즈 CSV 없음 - pose_extractor.py 먼저 실행")
    else:
        pose_df = pd.read_csv(CSV_PATH)
        fps = 59.6

        print("=" * 60)
        print("영상 구간 자동 추출 테스트")
        print("=" * 60)

        sliced_df = slice_pose_df(
            pose_df, fps,
            mode="pitch",
            pitcher_hand="L",  # Hader는 좌투
            min_detection_rate=0.7,
        )

        print(f"\n원본: {len(pose_df)}프레임 → 슬라이싱: {len(sliced_df)}프레임")
        print(f"시작: {sliced_df['time_sec'].min():.2f}초 / 끝: {sliced_df['time_sec'].max():.2f}초")
