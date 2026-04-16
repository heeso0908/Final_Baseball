"""
trimmed 영상 전체 배치 포즈 추출
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from pose_extractor_yolo import extract_pose_yolo

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trimmed_dir", default="videos/leiter/trimmed")
parser.add_argument("--output_dir",  default="pose_output/leiter")
_args = parser.parse_args()

TRIMMED_DIR = Path(__file__).parent / _args.trimmed_dir
OUTPUT_DIR  = Path(__file__).parent / _args.output_dir

videos = sorted(TRIMMED_DIR.glob("*_trimmed.mp4"))
print(f"대상: {len(videos)}개 영상\n")

ok, skip, fail = 0, 0, 0
for i, vpath in enumerate(videos):
    stem     = vpath.stem
    csv_path = OUTPUT_DIR / f"{stem}_yolo_pose.csv"

    if csv_path.exists():
        print(f"[{i+1}/{len(videos)}] SKIP (기존): {vpath.name}")
        skip += 1
        continue

    print(f"\n[{i+1}/{len(videos)}] {vpath.name}")
    try:
        extract_pose_yolo(
            video_path      = str(vpath),
            output_dir      = str(OUTPUT_DIR),
            save_debug_video= True,
        )
        ok += 1
    except Exception as e:
        print(f"  ✗ 오류: {e}")
        fail += 1

print(f"\n완료: 성공 {ok} | 스킵 {skip} | 실패 {fail}")
