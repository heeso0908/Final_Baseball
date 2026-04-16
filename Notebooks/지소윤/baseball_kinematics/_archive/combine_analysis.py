"""
전체 파이프라인 결합
Statcast 수집 → 영상 다운로드 → 포즈 추출 (YOLO) → 유형 분류 → 키네마틱 분석 → 결합
"""

import os
import time
import subprocess
import requests
import pandas as pd
import cv2

from pybaseball import statcast

from pose_extractor_yolo import extract_pose_yolo
from classifier import classify_and_analyze
from kinematics import analyze_full
from segment_extractor import slice_pose_df, check_classification_reliability

VIDEO_DIR  = "videos"
OUTPUT_DIR = "pose_output"


# ──────────────────────────────────────────
# 1. Statcast 수집
# ──────────────────────────────────────────

def get_statcast(start_date: str, end_date: str, pitcher_name: str) -> pd.DataFrame:
    print(f"\n[1] Statcast 수집: {pitcher_name} ({start_date} ~ {end_date})")
    df = statcast(start_dt=start_date, end_dt=end_date)
    df = df[df["player_name"].astype(str).str.contains(pitcher_name, case=False, na=False)]
    result = df[df["events"].notna()].copy()
    print(f"  → 타석 {len(result)}개")
    return result


# ──────────────────────────────────────────
# 2. play_id 수집
# ──────────────────────────────────────────

def get_play_ids(game_pk: int, ab_numbers: list) -> dict:
    url = f"https://baseballsavant.mlb.com/gf?game_pk={game_pk}"
    try:
        res  = requests.get(url, timeout=10)
        data = res.json()
        all_plays = data.get("team_home", []) + data.get("team_away", [])
        result = {}
        for play in all_plays:
            ab  = play.get("ab_number")
            pid = play.get("play_id")
            if ab in ab_numbers and play.get("events") and pid:
                result[ab] = pid
        return result
    except Exception as e:
        print(f"  [오류] game_pk {game_pk}: {e}")
        return {}


# ──────────────────────────────────────────
# 3. 영상 다운로드
# ──────────────────────────────────────────

def download_video(play_id: str, save_dir: str = VIDEO_DIR) -> str | None:
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/{play_id}.mp4"

    if os.path.exists(save_path):
        print(f"  이미 존재: {play_id}")
        return save_path

    url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    ret = subprocess.run(
        ["yt-dlp", url, "-o", save_path, "-q"],
        capture_output=True
    )
    if ret.returncode == 0 and os.path.exists(save_path):
        print(f"  ✓ 다운로드: {play_id}")
        return save_path
    else:
        print(f"  ✗ 실패: {play_id}")
        return None


# ──────────────────────────────────────────
# 4. 포즈 추출 (YOLO ByeTrack + MediaPipe)
# ──────────────────────────────────────────

def extract_pose(video_path: str) -> tuple[pd.DataFrame, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    # YOLO ByeTrack + MediaPipe
    # 첫 프레임에서 투수 클릭 창이 뜹니다
    df = extract_pose_yolo(
        video_path       = video_path,
        output_dir       = OUTPUT_DIR,
        save_debug_video = True,
        use_color_mask   = False,
        iou_threshold    = 0.3,
    )

    if df.empty:
        return df, fps

    detected_rate = df["detected"].mean() * 100
    print(f"  포즈 감지율: {detected_rate:.1f}% ({len(df)}프레임)")

    # 감지율 낮으면 경고
    if detected_rate < 85:
        print(f"  ⚠ 감지율 낮음 ({detected_rate:.1f}%) → 분석 정확도 낮을 수 있음")

    return df, fps


# ──────────────────────────────────────────
# 5. 전체 파이프라인
# ──────────────────────────────────────────

def run_pipeline(
    start_date:   str,
    end_date:     str,
    pitcher_name: str,
    mode:         str = "pitch",
    max_pitches:  int = 5,
    min_detection_rate: float = 0.7,
    save_dir:     str = OUTPUT_DIR,
) -> pd.DataFrame:

    os.makedirs(save_dir, exist_ok=True)

    # 1. Statcast
    statcast_df = get_statcast(start_date, end_date, pitcher_name)
    if statcast_df.empty:
        print("Statcast 데이터 없음")
        return pd.DataFrame()

    # 2. play_id
    print("\n[2] play_id 수집 중...")
    meta_rows = []
    for game_pk, group in statcast_df.groupby("game_pk"):
        ab_to_pid = get_play_ids(int(game_pk), group["at_bat_number"].tolist())
        for _, row in group.iterrows():
            pid = ab_to_pid.get(row["at_bat_number"])
            if pid:
                meta_rows.append({
                    "play_id":       pid,
                    "game_pk":       game_pk,
                    "game_date":     row["game_date"],
                    "pitcher_name":  row["player_name"],
                    "events":        row["events"],
                    "pitch_name":    row.get("pitch_name"),
                    "release_speed": row.get("release_speed"),
                    "spin_rate":     row.get("release_spin_rate"),
                    "launch_speed":  row.get("launch_speed"),
                    "launch_angle":  row.get("launch_angle"),
                    "estimated_ba":  row.get("estimated_ba_using_speedangle"),
                    "woba_value":    row.get("woba_value"),
                    "delta_run_exp": row.get("delta_run_exp"),
                    "zone":          row.get("zone"),
                })

    if not meta_rows:
        print("play_id 없음")
        return pd.DataFrame()

    meta_df = pd.DataFrame(meta_rows).head(max_pitches)
    print(f"  → {len(meta_df)}개 처리 예정")

    # 3~5. 영상 다운로드 → 포즈 추출 → 분석
    print("\n[3~5] 영상 다운로드 → 포즈 추출 → 분석...")
    analysis_rows = []

    for _, row in meta_df.iterrows():
        pid = row["play_id"]
        print(f"\n  처리: {pid} | {row['events']} | {row['release_speed']} mph")

        # 영상 다운로드
        video_path = download_video(pid)
        if not video_path:
            analysis_rows.append({})
            continue

        # 포즈 추출 (YOLO 클릭 창 뜸)
        pose_df, fps = extract_pose(video_path)
        if pose_df.empty or pose_df["detected"].sum() < 10:
            print("  ✗ 포즈 감지 실패 → 스킵")
            analysis_rows.append({})
            continue

        # 감지율 필터링
        if pose_df["detected"].mean() < min_detection_rate:
            print(f"  ✗ 감지율 미달 ({pose_df['detected'].mean()*100:.1f}%) → 스킵")
            analysis_rows.append({})
            continue

        # 유효 구간 추출
        pose_df = slice_pose_df(pose_df, fps, mode=mode, min_detection_rate=min_detection_rate)

        # 유형 분류 + 신뢰도 체크
        clf        = classify_and_analyze(pose_df, fps, mode=mode)
        clf        = check_classification_reliability(pose_df, clf)
        hand       = clf["hand"]
        thresholds = clf.get("thresholds")

        # 키네마틱 분석
        summary = analyze_full(pose_df, fps, hand=hand, mode=mode, thresholds=thresholds)
        summary["arm_slot"]        = clf["type"].get("arm_slot")
        summary["hand_classified"] = hand
        summary["detection_rate"]  = clf["type"].get("detection_rate")
        summary["clf_reliable"]    = clf["type"].get("overall_reliable")

        analysis_rows.append(summary)
        time.sleep(0.3)

    # 6. 결합 + 저장
    final_df = pd.concat(
        [meta_df.reset_index(drop=True), pd.DataFrame(analysis_rows).reset_index(drop=True)],
        axis=1
    )

    csv_path = f"{save_dir}/{pitcher_name.replace(', ', '_')}_combined.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"\n✓ 저장 완료: {csv_path}")

    # 결과 출력
    print("\n[결합 결과 요약]")
    view_cols = [c for c in [
        "game_date", "pitcher_name", "events", "pitch_name",
        "release_speed", "spin_rate", "launch_speed", "launch_angle",
        "arm_slot", "hand_classified", "clf_reliable",
        "ks_is_ptd", "ks_energy_build",
        "sway_detected", "early_open_pct",
        "knee_collapse_pct", "arm_flyout_pct",
        "late_release", "estimated_ba",
    ] if c in final_df.columns]
    print(final_df[view_cols].to_string())

    return final_df


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    df = run_pipeline(
        start_date   = "2024-07-01",
        end_date     = "2024-07-01",
        pitcher_name = "Hader",
        mode         = "pitch",
        max_pitches  = 4,
    )