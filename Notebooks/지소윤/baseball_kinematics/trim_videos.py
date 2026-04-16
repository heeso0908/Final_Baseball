"""
동영상 일괄 트림 — 투구 장면만 추출 [수정판]
=============================================
수정 내역:
  1. ffmpeg -c copy → re-encode 방식으로 변경
     (-c copy는 keyframe 기준으로만 잘려 time_sec 오차 발생)
  2. 정확한 시작점 보장: -ss를 -i 뒤로 이동 (입력 후 seek = frame-accurate)
  3. 트림 후 실제 duration 검증 추가
  4. 투수 감지 기준에 종횡비 조건 추가 (클로즈업 타자/코치 혼입 방지)

실행:
    python trim_videos.py --dir videos/leiter
    python trim_videos.py --video videos/leiter/xxx.mp4
    python trim_videos.py --dir videos/leiter --out videos/leiter/trimmed
"""

import os
import sys
import argparse
import subprocess
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


def get_video_duration(path: str) -> float:
    """ffprobe로 실제 duration 확인"""
    try:
        ret = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=10
        )
        return float(ret.stdout.strip())
    except Exception:
        return 0.0


def trim_one(
    video_path: str,
    out_dir: str,
    yolo_model,
    scan_interval: int = 3,
    min_pitcher_y: float = 0.45,
    min_run_sec: float = 1.5,
    pad_sec: float = 0.5,
    # [추가] 투수 박스 종횡비 필터 (너무 넓으면 클로즈업 타자일 가능성)
    max_aspect_ratio: float = 0.6,   # width/height < 0.6이어야 투수 (서있는 사람)
) -> str | None:
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    pitcher_frames = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % scan_interval == 0:
            res   = yolo_model.predict(frame, classes=[0], verbose=False, conf=0.4)
            boxes = res[0].boxes
            if boxes is not None and len(boxes.xyxy) > 0:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cy        = (y1 + y2) / 2 / h
                    box_w_pct = (x2 - x1) / fw
                    box_h_pct = (y2 - y1) / h
                    aspect    = (x2 - x1) / ((y2 - y1) + 1e-6)

                    # [수정] 종횡비 조건 추가: 서있는 사람은 세로가 더 길어야 함
                    if (cy >= min_pitcher_y
                            and box_w_pct > 0.06
                            and box_h_pct > 0.25       # 화면 높이의 25% 이상
                            and aspect < max_aspect_ratio):
                        pitcher_frames.append(frame_idx)
                        break
        frame_idx += 1

    cap.release()

    if not pitcher_frames:
        print(f"  ⚠ 투수 감지 없음 → 스킵")
        return None

    # 연속 구간 찾기
    segments, seg_start, prev = [], pitcher_frames[0], pitcher_frames[0]
    for f in pitcher_frames[1:]:
        if f - prev > scan_interval * 4:
            segments.append((seg_start, prev))
            seg_start = f
        prev = f
    segments.append((seg_start, prev))

    min_frames  = int(min_run_sec * fps)
    valid_segs  = [(s, e) for s, e in segments if e - s >= min_frames]
    if not valid_segs:
        print(f"  ⚠ 연속 구간 부족 → 스킵")
        return None

    best_start, best_end = max(valid_segs, key=lambda x: x[1] - x[0])
    t_start  = max(0.0,         (best_start / fps) - pad_sec)
    t_end    = min(total / fps, (best_end   / fps) + pad_sec)
    duration = t_end - t_start

    os.makedirs(out_dir, exist_ok=True)
    stem     = Path(video_path).stem
    out_path = str(Path(out_dir) / f"{stem}_trimmed.mp4")

    # [수정] frame-accurate trim:
    #   -ss를 -i 뒤에 배치 → keyframe 무관하게 정확한 프레임에서 자름
    #   libx264 re-encode → 타임스탬프 0부터 정확히 시작
    ret_code = subprocess.run([
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", f"{t_start:.3f}",      # [수정] -i 뒤로 이동
        "-t",  f"{duration:.3f}",
        "-c:v", "libx264",            # [수정] re-encode
        "-preset", "fast",
        "-crf", "18",                 # 고품질 (0=무손실, 51=최저화질)
        "-c:a", "aac",
        "-loglevel", "error",
        out_path,
    ]).returncode

    if ret_code != 0 or not os.path.exists(out_path):
        print(f"  ✗ ffmpeg 실패")
        return None

    # [추가] 실제 duration 검증
    actual_dur = get_video_duration(out_path)
    expected   = duration
    if abs(actual_dur - expected) > 0.5:
        print(f"  ⚠ duration 불일치: 예상 {expected:.1f}s / 실제 {actual_dur:.1f}s")

    orig_mb = os.path.getsize(video_path) / 1024 / 1024
    trim_mb = os.path.getsize(out_path)   / 1024 / 1024
    print(f"  ✓ {t_start:.1f}s ~ {t_end:.1f}s ({duration:.1f}초) | "
          f"{orig_mb:.1f}MB → {trim_mb:.1f}MB | 실제 {actual_dur:.1f}s")
    return out_path


def trim_dir(video_dir: str, out_dir: str, yolo_model):
    videos = sorted(Path(video_dir).glob("*.mp4"))
    videos = [v for v in videos if "_trimmed" not in v.name]

    print(f"대상: {len(videos)}개 영상")
    print(f"저장: {out_dir}\n")

    ok, skip = 0, 0
    for i, vpath in enumerate(videos):
        stem     = vpath.stem
        out_path = Path(out_dir) / f"{stem}_trimmed.mp4"

        # 이미 트림된 파일이 있고 유효하면 스킵
        if out_path.exists() and out_path.stat().st_size > 100_000:
            print(f"[{i+1}/{len(videos)}] SKIP (기존): {vpath.name}")
            skip += 1
            continue

        print(f"[{i+1}/{len(videos)}] {vpath.name}")
        result = trim_one(str(vpath), out_dir, yolo_model)
        if result:
            ok += 1
        else:
            skip += 1

    print(f"\n완료: 성공 {ok} | 스킵/실패 {skip}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",   help="영상 폴더 경로")
    parser.add_argument("--video", help="단일 영상 경로")
    parser.add_argument("--out",   default=None, help="출력 폴더")
    args = parser.parse_args()

    if not args.dir and not args.video:
        parser.print_help()
        sys.exit(1)

    print("YOLO 로딩...")
    yolo = YOLO("yolo11n.pt")

    if args.video:
        out_dir = args.out or str(Path(args.video).parent / "trimmed")
        print(f"\n{Path(args.video).name}")
        trim_one(args.video, out_dir, yolo)
    else:
        out_dir = args.out or str(Path(args.dir) / "trimmed")
        trim_dir(args.dir, out_dir, yolo)
