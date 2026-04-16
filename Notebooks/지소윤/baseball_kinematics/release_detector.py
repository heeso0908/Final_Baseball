"""
릴리즈 포인트 감지 [수정판]
============================
수정 내역:
  1. 카메라 뷰 방향 자동 추정 추가
     (센터필드뷰 vs 1루뷰에 따라 x_min/x_max 방향이 반대)
  2. velocity 방식: 단순 속도 피크 → pitch_phase 기반 탐색 구간 정제
     (와인드업 제외, 가속 구간 내 피크 탐색)
  3. 릴리즈 후보 복수 추출 후 합리성 검증 추가
     (팔꿈치 높이, 손목이 어깨 앞에 있는지 체크)
  4. search_end 0.85 → 0.80 (팔로우스루 혼입 방지 강화)
  5. 감지 실패 시 상세 원인 로깅

사용:
    from release_detector import detect_release, extract_window
    frame_idx, time_sec = detect_release(df, throw_hand="R")
    df_window = extract_window(df, frame_idx, pre=25, post=15)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


THROW_WRIST    = {"R": "right_wrist",    "L": "left_wrist"}
THROW_ELBOW    = {"R": "right_elbow",    "L": "left_elbow"}
THROW_SHOULDER = {"R": "right_shoulder", "L": "left_shoulder"}


# ──────────────────────────────────────────
# [추가] 카메라 뷰 방향 추정
# ──────────────────────────────────────────

def estimate_camera_side(df: pd.DataFrame, throw_hand: str = "R") -> str:
    """
    투구팔 어깨와 반대쪽 어깨의 x 관계로 카메라 뷰 방향 추정.

    센터필드뷰(우투): 투구팔(오른쪽) 어깨가 화면 오른쪽 → right_shoulder_x > left_shoulder_x
    1루뷰(우투):     투구팔(오른쪽) 어깨가 화면 왼쪽  → right_shoulder_x < left_shoulder_x

    반환: "cf" (센터필드) | "first" (1루측) | "unknown"
    """
    valid = df[df["detected"].fillna(False)].copy()
    if valid.empty:
        return "unknown"

    # 와인드업 ~ 스트라이드 구간 (앞 40%) 에서 평균 비교
    n   = len(valid)
    seg = valid.iloc[:int(n * 0.4)]
    if seg.empty:
        seg = valid

    if throw_hand == "R":
        throw_sh_col = "right_shoulder_x"
        other_sh_col = "left_shoulder_x"
    else:
        throw_sh_col = "left_shoulder_x"
        other_sh_col = "right_shoulder_x"

    if throw_sh_col not in seg or other_sh_col not in seg:
        return "unknown"

    throw_mean = seg[throw_sh_col].dropna().mean()
    other_mean = seg[other_sh_col].dropna().mean()

    if np.isnan(throw_mean) or np.isnan(other_mean):
        return "unknown"

    # 투구팔 어깨가 반대쪽보다 x가 크면 → 센터필드뷰
    return "cf" if throw_mean > other_mean else "first"


# ──────────────────────────────────────────
# 릴리즈 후보 합리성 검증
# ──────────────────────────────────────────

def validate_release_candidate(
    row: pd.Series,
    throw_hand: str = "R",
) -> bool:
    """
    릴리즈 후보 프레임이 실제 릴리즈 포인트로 타당한지 검증.
    조건:
      1. 손목이 어깨보다 앞에(x 기준) 있어야 함
      2. 팔꿈치가 어깨 높이 ± 30% 범위 내 (너무 아래면 팔로우스루)
    """
    wrist_col  = f"{THROW_WRIST[throw_hand]}_x"
    sh_col     = f"{THROW_SHOULDER[throw_hand]}_x"
    el_y_col   = f"{THROW_ELBOW[throw_hand]}_y"
    sh_y_col   = f"{THROW_SHOULDER[throw_hand]}_y"

    wrist_x = row.get(wrist_col)
    sh_x    = row.get(sh_col)
    el_y    = row.get(el_y_col)
    sh_y    = row.get(sh_y_col)

    # None/NaN 체크
    def valid_num(v):
        try:
            return v is not None and not np.isnan(float(v))
        except (TypeError, ValueError):
            return False

    # 조건 1: 손목이 어깨보다 앞 (우투=왼쪽 x값 작음, 좌투=반대)
    if valid_num(wrist_x) and valid_num(sh_x):
        if throw_hand == "R":
            cond1 = float(wrist_x) < float(sh_x) + 0.1   # 약간의 tolerance
        else:
            cond1 = float(wrist_x) > float(sh_x) - 0.1
    else:
        cond1 = True  # 정보 없으면 통과

    # 조건 2: 팔꿈치 높이가 어깨 ± 30% 범위 (y값은 아래로 갈수록 큰 값)
    if valid_num(el_y) and valid_num(sh_y):
        height_range = abs(float(el_y) - float(sh_y))
        cond2 = height_range < 0.30
    else:
        cond2 = True

    return cond1 and cond2


# ──────────────────────────────────────────
# 핵심: 릴리즈 포인트 감지
# ──────────────────────────────────────────

def detect_release(
    df: pd.DataFrame,
    throw_hand: str = "R",
    smooth_win: int = 5,
    method: str = "velocity",
    search_start: float = 0.10,
    search_end: float = 0.80,       # [수정] 0.85 → 0.80
    camera_side: str = "auto",      # [추가] "auto" | "cf" | "first"
) -> tuple:
    """
    릴리즈 포인트 감지.

    [수정] velocity 방식:
      - 단순 최대 피크 → find_peaks로 복수 후보 추출 후 합리성 검증
      - 합리성 통과 후보 중 가장 속도가 큰 것 선택

    [추가] camera_side: 카메라 방향에 따라 x_min/x_max 결정
    """
    wrist_col = f"{THROW_WRIST[throw_hand]}_x"
    wrist_y   = f"{THROW_WRIST[throw_hand]}_y"

    valid = df[df["detected"].fillna(False)].copy().reset_index(drop=True)
    if valid.empty or wrist_col not in valid.columns:
        print("    릴리즈 감지 실패: detected 프레임 없음")
        return None, None

    n  = len(valid)
    lo = int(n * search_start)
    hi = int(n * search_end)
    if hi <= lo + 5:
        hi = n
    valid_search = valid.iloc[lo:hi].copy()

    if valid_search.empty:
        print("    릴리즈 감지 실패: 탐색 구간 비어있음")
        return None, None

    wx = valid_search[wrist_col].values.astype(float)
    wy = valid_search[wrist_y].values.astype(float) if wrist_y in valid_search else np.zeros(len(wx))

    # NaN 선형 보간
    for arr in (wx, wy):
        nans = np.isnan(arr)
        if nans.all():
            print("    릴리즈 감지 실패: 손목 x/y 전부 NaN")
            return None, None
        if nans.any():
            idx_arr = np.arange(len(arr))
            arr[nans] = np.interp(idx_arr[nans], idx_arr[~nans], arr[~nans])

    if method == "velocity":
        dx = np.diff(wx, prepend=wx[0])
        dy = np.diff(wy, prepend=wy[0])
        speed    = np.sqrt(dx**2 + dy**2)
        speed_sm = uniform_filter1d(speed, size=smooth_win)

        # [수정] 단순 argmax → find_peaks로 복수 후보
        min_height = np.percentile(speed_sm, 60)   # 상위 40% 피크만
        peaks, props = find_peaks(speed_sm, height=min_height, distance=3)

        if len(peaks) == 0:
            # 피크 없으면 단순 argmax fallback
            peak_local = int(np.argmax(speed_sm))
            row = valid_search.iloc[peak_local]
            return int(row["frame"]), float(row["time_sec"])

        # [수정] 각 후보 합리성 검증
        peak_heights = props["peak_heights"]
        # 속도 높은 순으로 정렬
        sorted_idx = np.argsort(peak_heights)[::-1]

        for idx in sorted_idx:
            candidate_local = peaks[idx]
            candidate_row   = valid_search.iloc[candidate_local]
            if validate_release_candidate(candidate_row, throw_hand):
                return int(candidate_row["frame"]), float(candidate_row["time_sec"])

        # 합리성 통과 없으면 최대 속도 피크 사용
        peak_local = peaks[sorted_idx[0]]
        row = valid_search.iloc[peak_local]
        print("    ⚠ 합리성 검증 통과 후보 없음 → 최대 속도 피크 사용")
        return int(row["frame"]), float(row["time_sec"])

    else:
        # x_min / x_max 방식 (fallback)
        # [수정] 카메라 방향 자동 추정
        if camera_side == "auto":
            camera_side = estimate_camera_side(df, throw_hand)

        wx_sm = uniform_filter1d(wx, size=smooth_win)

        # 센터필드뷰에서 우투: 릴리즈 시 손목 x가 최솟값 (왼쪽으로 이동)
        # 1루뷰에서 우투: 릴리즈 시 손목 x가 최댓값 (오른쪽으로 이동)
        if throw_hand == "R":
            if camera_side in ("cf", "unknown"):
                peak_local = int(np.argmin(wx_sm))
            else:
                peak_local = int(np.argmax(wx_sm))
        else:
            if camera_side in ("cf", "unknown"):
                peak_local = int(np.argmax(wx_sm))
            else:
                peak_local = int(np.argmin(wx_sm))

        row = valid_search.iloc[peak_local]
        return int(row["frame"]), float(row["time_sec"])


def extract_window(
    df: pd.DataFrame,
    release_frame: int,
    pre: int  = 25,
    post: int = 15,
) -> pd.DataFrame:
    """릴리즈 프레임 기준 앞뒤 N프레임 추출."""
    mask = (df["frame"] >= release_frame - pre) & (df["frame"] <= release_frame + post)
    return df[mask].copy().reset_index(drop=True)


# ──────────────────────────────────────────
# 배치 감지
# ──────────────────────────────────────────

def detect_all(
    pose_dir: str,
    pattern: str = "*_trimmed_yolo_pose_proc.csv",
    throw_hand: str = "R",
    pre: int = 25,
    post: int = 15,
    save_windows: bool = True,
) -> pd.DataFrame:

    paths = sorted(Path(pose_dir).glob(pattern))
    records = []

    for i, fpath in enumerate(paths):
        play_id = fpath.stem.replace("_trimmed_yolo_pose_proc", "")
        df = pd.read_csv(fpath)

        print(f"[{i+1}/{len(paths)}] {play_id[:12]}...")

        # [수정] 카메라 뷰 자동 추정 후 velocity → x_fallback 순
        cam_side = estimate_camera_side(df, throw_hand)
        print(f"  카메라 뷰: {cam_side}")

        rel_frame, rel_time = detect_release(
            df, throw_hand=throw_hand, method="velocity", camera_side=cam_side
        )
        if rel_frame is None:
            rel_frame, rel_time = detect_release(
                df, throw_hand=throw_hand, method="x_min", camera_side=cam_side
            )

        if rel_frame is None:
            print(f"  ✗ 감지 실패")
            records.append({"play_id": play_id, "release_frame": None,
                            "release_time": None, "camera_side": cam_side})
            continue

        n_total = len(df)
        print(f"  릴리즈 frame={rel_frame} ({rel_time:.2f}s) / {n_total}프레임")

        if save_windows:
            win_df  = extract_window(df, rel_frame, pre=pre, post=post)
            out     = fpath.parent / f"{play_id}_release_window.csv"
            win_df.to_csv(out, index=False)

        records.append({
            "play_id":       play_id,
            "release_frame": rel_frame,
            "release_time":  rel_time,
            "total_frames":  n_total,
            "camera_side":   cam_side,
        })

    result   = pd.DataFrame(records)
    out_path = Path(pose_dir) / "release_points.csv"
    result.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
    return result


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",  default="pose_output/leiter")
    parser.add_argument("--hand", default="R")
    parser.add_argument("--pre",  type=int, default=25)
    parser.add_argument("--post", type=int, default=15)
    args = parser.parse_args()

    detect_all(args.dir, throw_hand=args.hand, pre=args.pre, post=args.post)
