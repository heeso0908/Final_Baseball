"""
Gemini Flash 자동 라벨링
- 무료 티어 사용 (분당 15회 제한)
- Google AI Studio API 키 필요 (무료)
  → https://aistudio.google.com/apikey

실행:
    pip install google-generativeai
    python auto_labeler.py --video videos/9266bb01-0c12-4b4e-a176-615ce3fe615a.mp4 --test
    python auto_labeler.py --video videos/9266bb01-0c12-4b4e-a176-615ce3fe615a.mp4
"""

import cv2
import base64
import json
import os
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # baseball_kinematics/.env 로드

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("google-generativeai 설치 필요: pip install google-generativeai")
    exit(1)


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

# .env 파일의 GEMINI_API_KEY 사용 (없으면 빈 문자열)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

PHASE_LABELS = {
    "windup":         0,
    "stride":         1,
    "cocking":        2,
    "acceleration":   3,
    "follow_through": 4,
    "other":         -1,
}

PROMPT = """You are a baseball biomechanics expert. Classify this pitching frame into one phase:

- windup: Both feet on ground, starting motion, weight on back foot
- stride: Lead leg lifted, moving toward home plate
- cocking: Lead foot landed, throwing arm at maximum external rotation (arm back and up)
- acceleration: Arm moving forward rapidly toward release point
- follow_through: After ball release, arm decelerating downward
- other: Not pitching (replay, crowd, close-up of ball, etc.)

Respond ONLY with valid JSON, no other text:
{"phase": "windup", "confidence": 0.9, "reason": "brief reason"}"""


# ──────────────────────────────────────────
# 1. Gemini 초기화
# ──────────────────────────────────────────

def init_gemini(api_key: str):
    client = genai.Client(api_key=api_key)
    print("Gemini 초기화 완료: gemini-2.0-flash-lite")
    return client


# ──────────────────────────────────────────
# 2. 프레임 추출
# ──────────────────────────────────────────

def extract_frames(
    video_path: str,
    sample_interval: int = 5,
    max_frames: int = 100,
) -> tuple[list, float]:
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"영상: {video_path}")
    print(f"  FPS: {fps:.1f} | 총 프레임: {total} | 샘플 간격: {sample_interval}프레임")

    frames    = []
    frame_idx = 0

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_interval == 0:
            frames.append((frame_idx, frame))
        frame_idx += 1

    cap.release()
    print(f"  → {len(frames)}개 프레임 추출")
    return frames, fps


def get_frame_at(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


# ──────────────────────────────────────────
# 3. Gemini Vision 분류
# ──────────────────────────────────────────

def classify_frame(client, frame: np.ndarray, frame_idx: int) -> dict:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(rgb)

    try:
        response = client.models.generate_content(
        model="gemini-2.0-flash-lite",  # ← 여기
        contents=[PROMPT, pil_img],
        )
        text = response.text.strip()

        if "```" in text:
            parts = text.split("```")
            for part in parts:
                if "{" in part:
                    text = part.strip()
                    if text.startswith("json"):
                        text = text[4:].strip()
                    break

        result = json.loads(text)
        result["label"] = PHASE_LABELS.get(result.get("phase", "other"), -1)
        return result

    except Exception as e:
        print(f"\n  [오류] {e}")
        return {"phase": "other", "confidence": 0.0, "reason": str(e)[:50], "label": -1}

# ──────────────────────────────────────────
# 4. 전체 자동 라벨링
# ──────────────────────────────────────────

def auto_label_video(
    video_path: str,
    client,
    sample_interval: int = 5,
    max_frames: int = 80,
    confidence_threshold: float = 0.7,
    output_dir: str = "pose_output",
) -> dict:

    frames, fps = extract_frames(video_path, sample_interval, max_frames)

    print(f"\nGemini Vision 분류 중... ({len(frames)}개 프레임)")
    print("(무료 티어: 분당 15회 제한 → 자동으로 속도 조절)\n")

    results  = []
    req_count = 0

    for i, (frame_idx, frame) in enumerate(frames):
        print(f"  [{i+1}/{len(frames)}] frame {frame_idx} ({frame_idx/fps:.2f}초)... ", end="", flush=True)

        result = classify_frame(client, frame, frame_idx)
        result["frame_idx"] = frame_idx
        result["time_sec"]  = frame_idx / fps
        results.append(result)

        phase = result["phase"]
        conf  = result["confidence"]
        print(f"{phase} ({conf:.0%}) | {result['reason'][:45]}")

        req_count += 1

        # 무료 티어 분당 15회 제한 → 12회마다 60초 대기
        if req_count % 12 == 0:
            print(f"\n  ⏳ API 제한 방지 대기 중 (60초)...\n")
            time.sleep(60)
        else:
            time.sleep(4.5)  # 60초 / 13회 ≈ 4.5초 간격

    # 결과 정리
    results_df = pd.DataFrame(results)

    # 보간: 샘플 프레임 사이도 같은 라벨
    labels = {}
    for _, row in results_df.iterrows():
        f    = int(row["frame_idx"])
        lbl  = int(row["label"])
        conf = float(row["confidence"])
        if conf >= confidence_threshold:
            for ff in range(f, f + sample_interval):
                labels[str(ff)] = lbl

    # 저장
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    label_path = os.path.join(output_dir, f"{video_name}_labels.json")
    csv_path   = os.path.join(output_dir, f"{video_name}_auto_label.csv")

    with open(label_path, "w") as f:
        json.dump(labels, f)
    results_df.to_csv(csv_path, index=False)

    # 요약
    print(f"\n[결과 요약]")
    for phase, count in results_df["phase"].value_counts().items():
        avg_conf = results_df[results_df["phase"] == phase]["confidence"].mean()
        print(f"  {phase}: {count}프레임 (평균 신뢰도 {avg_conf:.0%})")

    low_conf = results_df[results_df["confidence"] < confidence_threshold]
    if not low_conf.empty:
        print(f"\n  ⚠ 신뢰도 낮은 프레임: {len(low_conf)}개 → 라벨 미적용")

    print(f"\n✓ 라벨 저장: {label_path} ({len(labels)}프레임)")
    print(f"✓ 상세 결과: {csv_path}")

    return labels


# ──────────────────────────────────────────
# 5. 테스트 (10프레임)
# ──────────────────────────────────────────

def test_mode(video_path: str, model):
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    test_frames = [int(total * i / 10) for i in range(10)]
    print(f"테스트 프레임: {test_frames}")
    print(f"(총 {total}프레임, FPS {fps:.1f})\n")

    results = []
    for i, frame_idx in enumerate(test_frames):
        frame = get_frame_at(video_path, frame_idx)
        if frame is None:
            continue

        print(f"[{i+1}/10] frame {frame_idx} ({frame_idx/fps:.2f}초)... ", end="", flush=True)
        result = classify_frame(client, frame, frame_idx)
        result["frame_idx"] = frame_idx
        result["time_sec"]  = frame_idx / fps
        results.append(result)

        print(f"{result['phase']} ({result['confidence']:.0%}) | {result['reason'][:50]}")
        time.sleep(4.5)

    print("\n[테스트 결과]")
    df = pd.DataFrame(results)
    print(df[["frame_idx", "time_sec", "phase", "confidence", "reason"]].to_string(index=False))
    return df


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",    required=True)
    parser.add_argument("--test",     action="store_true")
    parser.add_argument("--interval", type=int,   default=5)
    parser.add_argument("--max",      type=int,   default=80)
    parser.add_argument("--conf",     type=float, default=0.7)
    parser.add_argument("--output",   default="pose_output")
    parser.add_argument("--apikey",   default=GEMINI_API_KEY)
    args = parser.parse_args()

    if args.apikey == "여기에_API_키_입력":
        print("❌ API 키를 입력하세요!")
        print("   1. https://aistudio.google.com/apikey 접속")
        print("   2. API 키 발급")
        print("   3. --apikey 옵션으로 전달:")
        print("      python auto_labeler.py --video ... --apikey YOUR_KEY")
        exit(1)

    client = init_gemini(args.apikey)

    if args.test:
        print("=" * 60)
        print("테스트 모드 (10프레임)")
        print("=" * 60)
        test_mode(args.video, client)
    else:
        print("=" * 60)
        print("전체 자동 라벨링")
        print("=" * 60)
        auto_label_video(
            video_path=args.video,
            client=client,
            sample_interval=args.interval,
            max_frames=args.max,
            confidence_threshold=args.conf,
            output_dir=args.output,
        )
