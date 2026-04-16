"""
포즈 CSV 전처리: 필터링 → 인터폴레이션 → 스무딩 [수정판]
=============================================================
수정 내역:
  1. NaN 포함 컬럼 스무딩 스킵 → 유효 구간만 분할 스무딩으로 변경
     (raw 관절 + 스무딩 관절 혼재 → 각도 점프 발생 문제 수정)
  2. visibility 필터를 관절별 독립 적용 (기존: 프레임 전체 무효)
  3. 전처리 품질 리포트 추가 (관절별 NaN 비율)
  4. critical joint NaN 비율이 높은 프레임 dropped 처리

사용:
    from pose_preprocessor import preprocess_pose
    df_clean = preprocess_pose("pose_output/leiter/xxx_yolo_pose.csv")
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import savgol_filter


JOINT_NAMES = [
    "nose",
    "left_shoulder",  "right_shoulder",
    "left_elbow",     "right_elbow",
    "left_wrist",     "right_wrist",
    "left_hip",       "right_hip",
    "left_knee",      "right_knee",
    "left_ankle",     "right_ankle",
]

XY_COLS  = [f"{j}_{ax}" for j in JOINT_NAMES for ax in ("x", "y")]
VIS_COLS = [f"{j}_vis" for j in JOINT_NAMES]

# 릴리즈 포인트 분석의 핵심 관절 (우투 기준)
CRITICAL_JOINTS = [
    "right_shoulder", "right_elbow", "right_wrist",
    "left_shoulder",
    "left_hip", "right_hip",
    "left_knee", "left_ankle",
]


# ── 1. 프레임 레벨 필터링 ──────────────────────────────────────

def filter_frames(df: pd.DataFrame, vis_thresh: float = 0.5) -> pd.DataFrame:
    """
    - detected=False 또는 invalid_reason 있는 프레임 → 전체 관절 NaN
    - [수정] visibility 낮은 관절 → 해당 관절만 NaN (프레임 전체 무효 X)
    - [수정] vis_thresh 기본값 0.3 → 0.5 (수집 단계 confidence와 통일)
    """
    df = df.copy()

    bad_frame = ~df["detected"].fillna(False)
    if "invalid_reason" in df.columns:
        bad_frame |= df["invalid_reason"].fillna("").str.len() > 0

    # 배드 프레임: 전체 관절 NaN
    for jname in JOINT_NAMES:
        for ax in ("x", "y"):
            col = f"{jname}_{ax}"
            if col in df.columns:
                df.loc[bad_frame, col] = np.nan

    # [수정] visibility 낮은 관절만 개별 NaN (프레임 전체 X)
    for jname in JOINT_NAMES:
        vis_col = f"{jname}_vis"
        if vis_col not in df.columns:
            continue
        low_vis = df[vis_col].fillna(0) < vis_thresh
        # 배드 프레임에서는 이미 NaN이므로 low_vis만 추가 처리
        target = (~bad_frame) & low_vis
        for ax in ("x", "y"):
            col = f"{jname}_{ax}"
            if col in df.columns:
                df.loc[target, col] = np.nan

    return df


# ── 2. 인터폴레이션 ────────────────────────────────────────────

def interpolate_joints(df: pd.DataFrame, max_gap: int = 5) -> pd.DataFrame:
    """
    NaN 관절 좌표를 linear interpolation.
    연속 NaN 구간이 max_gap 프레임 초과하면 그대로 NaN 유지.
    """
    df = df.copy()
    for col in XY_COLS:
        if col not in df.columns:
            continue
        s      = df[col].copy()
        is_nan = s.isna()
        if not is_nan.any():
            continue

        gap_ids = (is_nan != is_nan.shift()).cumsum()
        gap_len = is_nan.groupby(gap_ids).transform("sum")
        can_interp = is_nan & (gap_len <= max_gap)

        s_interp = s.interpolate(method="linear", limit_direction="both")
        s[can_interp] = s_interp[can_interp]
        df[col] = s
    return df


# ── 3. 스무딩 [핵심 수정] ─────────────────────────────────────

def smooth_joints(df: pd.DataFrame, window: int = 7, poly: int = 2) -> pd.DataFrame:
    """
    Savitzky-Golay 필터로 관절 좌표 스무딩.

    [수정] 기존: NaN 있으면 컬럼 전체 스킵 → raw/스무딩 혼재 → 각도 점프
    [수정 후]: NaN으로 분리된 연속 유효 구간을 각각 독립 스무딩

    즉, NaN 구간으로 분리된 각 valid 세그먼트에 개별적으로
    savgol_filter를 적용하여 세그먼트 간 불연속 없이 처리.
    세그먼트가 window보다 짧으면 window를 세그먼트 길이에 맞게 줄임.
    """
    if window % 2 == 0:
        window += 1

    df = df.copy()

    for col in XY_COLS:
        if col not in df.columns:
            continue

        s      = df[col].copy().values.astype(float)
        is_nan = np.isnan(s)

        if not is_nan.any():
            # NaN 없으면 전체 스무딩
            if len(s) >= window:
                df[col] = savgol_filter(s, window_length=window, polyorder=poly)
            continue

        if is_nan.all():
            continue

        # [수정] NaN으로 나뉜 유효 구간을 각각 스무딩
        smoothed = s.copy()
        in_seg   = False
        seg_start = 0

        for i in range(len(s) + 1):
            currently_valid = (i < len(s)) and not is_nan[i]

            if currently_valid and not in_seg:
                seg_start = i
                in_seg    = True
            elif not currently_valid and in_seg:
                seg_end = i  # exclusive
                seg     = s[seg_start:seg_end]
                seg_len = len(seg)

                if seg_len >= 4:   # 최소 4프레임 이상이어야 스무딩 의미 있음
                    # 세그먼트 길이에 맞게 window 조정 (홀수 유지)
                    win = min(window, seg_len)
                    if win % 2 == 0:
                        win -= 1
                    if win >= 3:
                        smoothed[seg_start:seg_end] = savgol_filter(
                            seg, window_length=win, polyorder=min(poly, win-1)
                        )
                in_seg = False

        df[col] = smoothed

    return df


# ── 4. 품질 리포트 ─────────────────────────────────────────────

def quality_report(df: pd.DataFrame) -> dict:
    """
    전처리 후 관절별 NaN 비율 + critical joint 품질 체크
    """
    report = {}
    total  = len(df)
    detected = df["detected"].fillna(False).sum()
    report["total_frames"]    = total
    report["detected_frames"] = int(detected)
    report["detection_rate"]  = round(float(detected / total), 3) if total > 0 else 0

    joint_nan = {}
    for jname in JOINT_NAMES:
        col = f"{jname}_x"
        if col in df.columns:
            nan_rate = df[col].isna().mean()
            joint_nan[jname] = round(float(nan_rate), 3)
    report["joint_nan_rate"] = joint_nan

    # critical joint 평균 NaN 비율
    crit_nan = np.mean([joint_nan.get(j, 1.0) for j in CRITICAL_JOINTS])
    report["critical_nan_rate"] = round(float(crit_nan), 3)
    report["usable"] = crit_nan < 0.4   # critical joint 40% 이상 유효해야 분석 가능

    return report


# ── 5. 통합 파이프라인 ─────────────────────────────────────────

def preprocess_pose(
    csv_path: str,
    vis_thresh: float = 0.5,      # [수정] 0.3 → 0.5
    max_gap: int = 5,
    smooth_window: int = 7,
    smooth_poly: int = 2,
    save: bool = True,
    out_suffix: str = "_proc",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    단일 CSV 전처리 파이프라인.
    save=True이면 같은 디렉토리에 {stem}{out_suffix}.csv 저장.
    """
    df    = pd.read_csv(csv_path)
    n_raw = len(df)

    df = filter_frames(df,         vis_thresh=vis_thresh)
    df = interpolate_joints(df,    max_gap=max_gap)
    df = smooth_joints(df,         window=smooth_window, poly=smooth_poly)

    report = quality_report(df)

    if verbose:
        print(f"  감지율: {report['detection_rate']:.1%} "
              f"({report['detected_frames']}/{report['total_frames']})")
        print(f"  critical joint NaN: {report['critical_nan_rate']:.1%} "
              f"| 분석 가능: {'✓' if report['usable'] else '✗'}")

        # 문제 관절 출력
        bad_joints = [j for j, r in report["joint_nan_rate"].items() if r > 0.3]
        if bad_joints:
            print(f"  NaN>30% 관절: {', '.join(bad_joints)}")

    if save:
        stem     = Path(csv_path).stem
        out_path = Path(csv_path).parent / f"{stem}{out_suffix}.csv"
        df.to_csv(out_path, index=False)
        if verbose:
            print(f"  저장: {out_path.name}")

    return df


# ── 6. 배치 처리 ───────────────────────────────────────────────

def preprocess_all(
    input_dir: str,
    pattern: str = "*_yolo_pose.csv",
    out_suffix: str = "_proc",
    skip_unusable: bool = True,
    **kwargs,
) -> dict:
    paths = sorted(Path(input_dir).glob(pattern))
    paths = [p for p in paths if out_suffix not in p.stem]

    print(f"대상: {len(paths)}개\n")
    results  = {}
    skipped  = []

    for i, p in enumerate(paths):
        print(f"[{i+1}/{len(paths)}] {p.name}")
        df = preprocess_pose(str(p), out_suffix=out_suffix, **kwargs)

        report = quality_report(df)
        if skip_unusable and not report["usable"]:
            print(f"  → 품질 미달 (critical NaN {report['critical_nan_rate']:.0%}) — 스킵")
            skipped.append(p.stem)
            continue

        results[p.stem] = df

    print(f"\n완료: {len(results)}개 처리 | {len(skipped)}개 품질 미달 스킵")
    if skipped:
        print(f"  스킵: {skipped}")
    return results


# ── 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir",    help="CSV 폴더 (배치)")
    parser.add_argument("--csv",    help="단일 CSV 경로")
    parser.add_argument("--vis",    type=float, default=0.5)
    parser.add_argument("--gap",    type=int,   default=5)
    parser.add_argument("--window", type=int,   default=7)
    args = parser.parse_args()

    kw = dict(vis_thresh=args.vis, max_gap=args.gap, smooth_window=args.window)

    if args.csv:
        preprocess_pose(args.csv, **kw)
    elif args.dir:
        preprocess_all(args.dir, **kw)
    else:
        parser.print_help()
