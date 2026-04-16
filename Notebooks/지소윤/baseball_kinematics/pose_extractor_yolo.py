"""
YOLO ByteTrack + MediaPipe 투수 트래킹 [수정판]
================================================
수정 내역:
  1. 좌표 역변환 버그 수정 (패딩 포함 크롭 → 원본 좌표 변환)
  2. MediaPipe confidence 0.3 → 0.5 (엉뚱한 관절 혼입 방지)
  3. detected 판정: 절반 → 핵심 관절(투구팔+하체) 기준
  4. 비대칭 패딩 → 좌우 대칭 패딩 (투구팔 관절 누락 방지)
  5. max_y_drift 0.25 → 0.15 (박스 스냅 오류 감소)
  6. 박스 아웃 바운드 체크: 정규화 좌표 기준으로 통일
  7. 관절별 visibility 기록을 detected 판정과 분리
"""

import cv2
import numpy as np
import pandas as pd
import argparse
import os
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode
from ultralytics import YOLO

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker.task")

LANDMARKS = {
    "nose":            0,
    "left_shoulder":  11, "right_shoulder": 12,
    "left_elbow":     13, "right_elbow":    14,
    "left_wrist":     15, "right_wrist":    16,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
}

# 투구 분석 핵심 관절 — 이 중 충분한 수가 있어야 detected=True
# 우투 기준; 좌투는 run 시 throw_hand="L" 전달로 처리
CRITICAL_JOINTS_R = [
    "right_shoulder", "right_elbow", "right_wrist",   # 투구팔
    "left_shoulder",                                    # 반대쪽 어깨
    "left_hip", "right_hip",                           # 골반
    "left_knee", "left_ankle",                         # 앞발 (우투=왼발)
]
CRITICAL_JOINTS_L = [
    "left_shoulder", "left_elbow", "left_wrist",
    "right_shoulder",
    "left_hip", "right_hip",
    "right_knee", "right_ankle",
]

CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow",    "right_wrist"),
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),
    ("left_knee",      "left_ankle"),
    ("right_hip",      "right_knee"),
    ("right_knee",     "right_ankle"),
]


# ──────────────────────────────────────────
# IoU
# ──────────────────────────────────────────

def calc_iou(box1: list, box2: list) -> float:
    x1 = max(box1[0], box2[0]);  y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]);  y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    a2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    return float(inter / (a1 + a2 - inter + 1e-6))


# ──────────────────────────────────────────
# 1. 첫 프레임 클릭으로 투수 track_id 선택
# ──────────────────────────────────────────

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


def select_pitcher_track_id(frame: np.ndarray, yolo_model) -> tuple | None:
    global click_point
    click_point = None
    h, w = frame.shape[:2]

    results = yolo_model.track(frame, classes=[0], tracker="bytetrack.yaml",
                                persist=True, verbose=False)

    boxes_with_id = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes
        if boxes.id is not None:
            for box, track_id, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.id.cpu().numpy(),
                boxes.conf.cpu().numpy()
            ):
                if conf > 0.4:
                    boxes_with_id.append({"box": box.tolist(), "id": int(track_id)})

    display = frame.copy()
    for item in boxes_with_id:
        x1, y1, x2, y2 = [int(v) for v in item["box"]]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, f"ID:{item['id']}",
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display, "투수를 클릭하세요",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(display, "ESC = 취소",
                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)

    scale = min(1.0, 1280 / w, 720 / h)
    disp  = cv2.resize(display, (int(w * scale), int(h * scale)))
    cv2.namedWindow("투수 선택", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("투수 선택", mouse_callback)
    cv2.imshow("투수 선택", disp)

    print(f"\n  [투수 선택] 첫 프레임에서 투수를 클릭하세요...")
    print(f"  감지된 사람: {len(boxes_with_id)}명 (ID: {[b['id'] for b in boxes_with_id]})")

    while True:
        key = cv2.waitKey(50)
        if key == 27:
            cv2.destroyAllWindows()
            return None
        if click_point is not None:
            break

    cv2.destroyAllWindows()
    orig_x = int(click_point[0] / scale)
    orig_y = int(click_point[1] / scale)

    if not boxes_with_id:
        print("  감지된 사람 없음")
        return None

    best = min(boxes_with_id,
               key=lambda item: np.hypot(
                   (item["box"][0]+item["box"][2])/2 - orig_x,
                   (item["box"][1]+item["box"][3])/2 - orig_y))
    print(f"  선택된 투수 track_id: {best['id']} | 박스: {[int(v) for v in best['box']]}")
    return best["id"], best["box"]


# ──────────────────────────────────────────
# 2. track_id로 투수 박스 추적 + IoU 겹침 감지
# ──────────────────────────────────────────

def get_pitcher_box_by_id(
    frame: np.ndarray,
    yolo_model,
    pitcher_id: int,
    prev_box: list,
    iou_threshold: float = 0.3,
    # [수정] 0.25 → 0.15: 박스 스냅 오류 감소
    max_y_drift: float = 0.15,
) -> tuple:
    h = frame.shape[0]
    results = yolo_model.track(frame, classes=[0], tracker="bytetrack.yaml",
                                persist=True, verbose=False)

    if not results or results[0].boxes is None:
        return prev_box, 0.0, []

    boxes = results[0].boxes
    if boxes.id is None:
        return prev_box, 0.0, []

    pitcher_box = None
    other_boxes = []

    for box, track_id, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.id.cpu().numpy(),
        boxes.conf.cpu().numpy()
    ):
        if conf < 0.3:
            continue
        if int(track_id) == pitcher_id:
            pitcher_box = box.tolist()
        else:
            other_boxes.append(box.tolist())

    # [수정] y드리프트 체크 + 박스 면적 급변 체크 추가
    if pitcher_box is not None and prev_box is not None:
        prev_cy = (prev_box[1] + prev_box[3]) / 2 / h
        curr_cy = (pitcher_box[1] + pitcher_box[3]) / 2 / h
        # 면적 비율 체크 (3배 이상 차이나면 오감지)
        prev_area = (prev_box[2]-prev_box[0]) * (prev_box[3]-prev_box[1])
        curr_area = (pitcher_box[2]-pitcher_box[0]) * (pitcher_box[3]-pitcher_box[1])
        area_ratio = curr_area / (prev_area + 1e-6)

        if abs(curr_cy - prev_cy) > max_y_drift or not (0.33 < area_ratio < 3.0):
            pitcher_box = prev_box  # 위치/크기 튐 → 이전 박스 유지

    if pitcher_box is None:
        pitcher_box = prev_box

    max_iou = 0.0
    if other_boxes and pitcher_box:
        max_iou = max(calc_iou(pitcher_box, b) for b in other_boxes)

    return pitcher_box, max_iou, other_boxes


# ──────────────────────────────────────────
# 3. 크롭 + MediaPipe 포즈 추출 [핵심 수정]
# ──────────────────────────────────────────

def apply_white_mask(crop: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 60, 255]))
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=2)
    masked = crop.copy()
    masked[mask == 0] = [80, 80, 80]
    return masked


def extract_pose_in_box(
    frame: np.ndarray,
    box: list,
    landmarker,
    frame_idx: int,
    fps: float,
    # [수정] 좌우 대칭 패딩 (기존: 왼쪽만 pad_x_left, 오른쪽 0)
    padding: float = 0.2,
    use_color_mask: bool = True,
    throw_hand: str = "R",
) -> dict:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box

    # [수정] 좌우 대칭 패딩
    pad_x = (x2 - x1) * padding
    pad_y = (y2 - y1) * padding

    x1c = max(0, x1 - pad_x)
    y1c = max(0, y1 - pad_y)
    x2c = min(w, x2 + pad_x)    # [수정] 오른쪽도 패딩 적용
    y2c = min(h, y2 + pad_y)

    # 최소 너비 보장
    crop_w_now = x2c - x1c
    crop_h_now = y2c - y1c
    if crop_w_now < crop_h_now * 0.6:
        expand = (crop_h_now * 0.6 - crop_w_now) / 2
        x1c = max(0, x1c - expand)
        x2c = min(w, x2c + expand)

    crop = frame[int(y1c):int(y2c), int(x1c):int(x2c)]
    if crop.size == 0:
        return None

    if use_color_mask:
        crop = apply_white_mask(crop)

    crop_h, crop_w = crop.shape[:2]
    rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    ts_ms  = int(frame_idx * 1000 / fps)
    result = landmarker.detect_for_video(mp_img, ts_ms)

    row = {"frame": frame_idx, "time_sec": frame_idx / fps}

    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        critical = CRITICAL_JOINTS_R if throw_hand == "R" else CRITICAL_JOINTS_L
        critical_valid = 0

        for name, idx in LANDMARKS.items():
            lm = result.pose_landmarks[0][idx]

            # [수정] 크롭 기준 정규화 좌표 → 원본 픽셀 → 원본 정규화로 변환
            # lm.x, lm.y는 크롭 이미지 기준 0~1
            # 원본 픽셀 좌표
            px = lm.x * crop_w + x1c   # 픽셀
            py = lm.y * crop_h + y1c   # 픽셀
            # 원본 이미지 기준 정규화
            orig_x = px / w
            orig_y = py / h

            # [수정] 박스 아웃 바운드 체크: 정규화 좌표로 통일
            box_x1_norm = x1c / w
            box_y1_norm = y1c / h
            box_x2_norm = x2c / w
            box_y2_norm = y2c / h

            if (orig_x < box_x1_norm or orig_x > box_x2_norm or
                    orig_y < box_y1_norm or orig_y > box_y2_norm):
                row[f"{name}_x"]   = None
                row[f"{name}_y"]   = None
            else:
                row[f"{name}_x"]   = round(orig_x, 6)
                row[f"{name}_y"]   = round(orig_y, 6)
                if name in critical:
                    critical_valid += 1

            row[f"{name}_z"]   = round(float(lm.z), 6)
            row[f"{name}_vis"] = round(float(lm.visibility), 4)

        # [수정] detected 판정: 핵심 관절 기준 (전체 절반 → 핵심 관절 75% 이상)
        row["detected"] = critical_valid >= int(len(critical) * 0.75)
        row["critical_valid"] = critical_valid
        row["critical_total"] = len(critical)

    else:
        for name in LANDMARKS:
            row[f"{name}_x"] = row[f"{name}_y"] = None
            row[f"{name}_z"] = row[f"{name}_vis"] = None
        row["detected"]        = False
        row["critical_valid"]  = 0
        row["critical_total"]  = len(CRITICAL_JOINTS_R)

    row["box_x1"] = round(x1c / w, 6)
    row["box_y1"] = round(y1c / h, 6)
    row["box_x2"] = round(x2c / w, 6)
    row["box_y2"] = round(y2c / h, 6)
    return row


# ──────────────────────────────────────────
# 4. 자동 트림
# ──────────────────────────────────────────

def auto_trim_video(
    video_path: str,
    output_dir: str = "pose_output",
    yolo_model=None,
    scan_interval: int = 3,
    min_pitcher_y: float = 0.45,
    min_run_sec: float = 1.5,
    pad_sec: float = 0.4,
) -> str:
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fw    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    if yolo_model is None:
        yolo_model = YOLO("yolo11n.pt")

    print(f"  [자동 트림] YOLO 스캔 중... ({total}프레임, {total/fps:.1f}초)")

    pitcher_frames = []
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % scan_interval == 0:
            res = yolo_model.predict(frame, classes=[0], verbose=False, conf=0.4)
            boxes = res[0].boxes
            if boxes is not None and len(boxes.xyxy) > 0:
                for box in boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = box
                    cy    = (y1 + y2) / 2 / h
                    box_w = (x2 - x1) / fw
                    if cy >= min_pitcher_y and box_w > 0.08:
                        pitcher_frames.append(frame_idx)
                        break
        frame_idx += 1
    cap.release()

    if not pitcher_frames:
        print("  ⚠ 투수 감지 없음 → 원본 사용")
        return video_path

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
        print("  ⚠ 충분한 연속 구간 없음 → 원본 사용")
        return video_path

    best_start, best_end = max(valid_segs, key=lambda x: x[1] - x[0])
    t_start  = max(0.0,           (best_start / fps) - pad_sec)
    t_end    = min(total / fps,   (best_end   / fps) + pad_sec)
    duration = t_end - t_start
    print(f"  투구 구간: {t_start:.2f}s ~ {t_end:.2f}s ({duration:.2f}초)")

    os.makedirs(output_dir, exist_ok=True)
    base     = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{base}_trimmed.mp4")

    import subprocess as _sp
    # [유지] -c copy 는 속도 이점이 있지만 타임스탬프 오차 주의
    # trim_videos.py 에서 re-encode 로 처리하므로 여기선 copy 유지
    ret_code = _sp.run([
        "ffmpeg", "-y",
        "-ss", str(t_start), "-i", video_path,
        "-t", str(duration),
        "-c", "copy", "-loglevel", "error",
        out_path,
    ]).returncode

    if ret_code == 0 and os.path.exists(out_path):
        size_mb = os.path.getsize(out_path) / 1024 / 1024
        print(f"  ✓ 트림 완료: {os.path.basename(out_path)} ({size_mb:.1f} MB)")
        return out_path
    else:
        print("  ⚠ ffmpeg 트림 실패 → 원본 사용")
        return video_path


# ──────────────────────────────────────────
# 5. 전체 파이프라인
# ──────────────────────────────────────────

def extract_pose_yolo(
    video_path: str,
    output_dir: str = "pose_output",
    save_debug_video: bool = True,
    use_color_mask: bool = True,
    iou_threshold: float = 0.3,
    trim: bool = False,
    throw_hand: str = "R",       # [추가] 투구손 명시적 전달
) -> pd.DataFrame:

    os.makedirs(output_dir, exist_ok=True)

    print("YOLO 모델 로딩...")
    yolo = YOLO("yolo11n.pt")

    if trim:
        video_path = auto_trim_video(video_path, output_dir=output_dir, yolo_model=yolo)

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"영상: {video_path}")
    print(f"  {w}x{h} | {fps:.1f}fps | {total}프레임 ({total/fps:.1f}초)")

    ret, first_frame = cap.read()
    if not ret:
        print("영상을 읽을 수 없어요")
        return pd.DataFrame()

    result = select_pitcher_track_id(first_frame, yolo)
    if result is None:
        print("투수 선택 취소")
        return pd.DataFrame()

    pitcher_id, pitcher_box = result
    print(f"  투수 track_id: {pitcher_id} | 투구손: {throw_hand}")
    print(f"  IoU 겹침 임계값: {iou_threshold}")

    # [수정] confidence 0.3 → 0.5
    options = PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    if save_debug_video:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        debug_path = os.path.join(output_dir, f"{video_name}_yolo_debug.mp4")
        out_video  = cv2.VideoWriter(debug_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    rows, prev_box, overlap_count = [], pitcher_box, 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx == 0:
                box, max_iou = pitcher_box, 0.0
            else:
                box, max_iou, _ = get_pitcher_box_by_id(
                    frame, yolo, pitcher_id, prev_box, iou_threshold
                )

            is_overlap = max_iou > iou_threshold

            if is_overlap:
                overlap_count += 1
                row = {
                    "frame": frame_idx, "time_sec": frame_idx / fps,
                    "detected": False,
                    "invalid_reason": f"overlap(iou={max_iou:.2f})",
                    "max_iou": round(max_iou, 3),
                    "critical_valid": 0, "critical_total": len(CRITICAL_JOINTS_R),
                }
                for name in LANDMARKS:
                    row[f"{name}_x"] = row[f"{name}_y"] = None
                    row[f"{name}_z"] = row[f"{name}_vis"] = None
                row["box_x1"] = row["box_y1"] = row["box_x2"] = row["box_y2"] = None
            else:
                if box is not None:
                    prev_box = box
                row = extract_pose_in_box(
                    frame, box, landmarker, frame_idx, fps,
                    use_color_mask=use_color_mask,
                    throw_hand=throw_hand,          # [수정] throw_hand 전달
                )
                if row:
                    row["invalid_reason"] = ""
                    row["max_iou"] = round(max_iou, 3)

            if row:
                rows.append(row)

            # 디버그 영상
            if save_debug_video and row:
                debug_frame = frame.copy()
                if box is not None:
                    bx1, by1, bx2, by2 = [int(v) for v in box]
                    color = (0, 0, 255) if is_overlap else (0, 255, 0)
                    cv2.rectangle(debug_frame, (bx1, by1), (bx2, by2), color, 2)
                    label = (f"OVERLAP IoU:{max_iou:.2f}" if is_overlap
                             else f"PITCHER ID:{pitcher_id} "
                                  f"crit:{row.get('critical_valid',0)}/{row.get('critical_total',0)}")
                    cv2.putText(debug_frame, label, (bx1, by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                if row.get("detected") and not row.get("invalid_reason"):
                    pts = {}
                    for name in LANDMARKS:
                        xv = row.get(f"{name}_x")
                        yv = row.get(f"{name}_y")
                        if xv is not None and yv is not None:
                            pts[name] = (int(xv * w), int(yv * h))
                    for a, b in CONNECTIONS:
                        if a in pts and b in pts:
                            cv2.line(debug_frame, pts[a], pts[b], (0, 255, 255), 2)
                    for name, pt in pts.items():
                        cv2.circle(debug_frame, pt, 5, (0, 0, 255), -1)

                out_video.write(debug_frame)

            frame_idx += 1
            if frame_idx % 60 == 0:
                print(f"  {frame_idx}/{total} ({frame_idx/total*100:.0f}%) | 겹침: {overlap_count}")

    cap.release()
    if save_debug_video:
        out_video.release()
        print(f"  디버그 영상: {debug_path}")

    df = pd.DataFrame(rows)
    detected_rate = df["detected"].mean() * 100
    print(f"  포즈 감지율: {detected_rate:.1f}%")
    print(f"  겹침 제외: {overlap_count}프레임 ({overlap_count/total*100:.1f}%)")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    csv_path   = os.path.join(output_dir, f"{video_name}_yolo_pose.csv")
    df.to_csv(csv_path, index=False)
    print(f"  CSV 저장: {csv_path}")
    return df


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     default="pose_output")
    parser.add_argument("--no_debug",   action="store_true")
    parser.add_argument("--no_mask",    action="store_true")
    parser.add_argument("--iou",        type=float, default=0.3)
    parser.add_argument("--trim",       action="store_true")
    parser.add_argument("--throw_hand", default="R", choices=["R", "L"])
    args = parser.parse_args()

    df = extract_pose_yolo(
        video_path       = args.video,
        output_dir       = args.output,
        save_debug_video = not args.no_debug,
        use_color_mask   = not args.no_mask,
        iou_threshold    = args.iou,
        trim             = args.trim,
        throw_hand       = args.throw_hand,
    )
    print(f"\n완료! 총 {len(df)}프레임")
    print(f"감지율: {df['detected'].mean()*100:.1f}%")
