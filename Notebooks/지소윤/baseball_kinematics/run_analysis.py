"""
포즈 분석 파이프라인 [수정판]
================================
수정 내역:
  1. 어깨/골반 기울기 atan2 wraparound 수정
     → calc_kinematics() 후 np.unwrap() 일괄 적용
  2. 팔꿈치 각도 유효 구간 제한
     → 릴리즈 기준 -15~0프레임만 사용 (gimbal lock 혼입 방지)
  3. early_open 플래그: unwrap 후 각도로 재계산
  4. POST_FRAMES 15 → 10 (릴리즈 이후 무릎 오감지 구간 축소)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.ndimage import uniform_filter1d

sys.path.insert(0, str(Path(__file__).parent))
from release_detector import detect_release, extract_window

import argparse as _ap
_parser = _ap.ArgumentParser()
_parser.add_argument("--pose_dir", default="pose_output/leiter")
_parser.add_argument("--meta_csv", default=None)
_parser.add_argument("--out_csv",  default=None)
_parser.add_argument("--label",    default="walks", help="walks | so")
_args, _ = _parser.parse_known_args()

POSE_DIR  = Path(__file__).parent / _args.pose_dir
META_CSV  = Path(__file__).parent / (_args.meta_csv or f"pose_output/leiter/leiter_{_args.label}_downloaded.csv")
OUT_CSV   = Path(__file__).parent / (_args.out_csv  or f"{POSE_DIR}/leiter_{_args.label}_analysis.csv")

THROW_HAND  = "R"
NAN_THRESH  = 0.5
MIN_FRAMES  = 15
PRE_FRAMES  = 25
POST_FRAMES = 10    # [수정] 15 → 10: 릴리즈 이후 오감지 구간 축소

# 팔꿈치 각도 유효 구간: 릴리즈 기준 프레임 오프셋
ELBOW_VALID_PRE  = 15   # 릴리즈 15프레임 전부터
ELBOW_VALID_POST = 0    # 릴리즈까지만 (이후 gimbal lock 가능성)


# ── 계산 유틸 ──────────────────────────────────────────────────

def calc_angle(a, b, c) -> float:
    a, b, c = np.array(a, dtype=float), np.array(b, dtype=float), np.array(c, dtype=float)
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        return np.nan
    ba, bc = a - b, c - b
    denom  = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom < 1e-6:
        return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)))


VIS_THRESH = 0.7


def pt(row, name, vis_thresh=VIS_THRESH):
    x   = row.get(f"{name}_x")
    y   = row.get(f"{name}_y")
    vis = row.get(f"{name}_vis")
    try:
        if vis is not None and float(vis) < vis_thresh:
            return (np.nan, np.nan)
    except (TypeError, ValueError):
        pass
    return (x, y)


def is_valid(*points):
    for p in points:
        for v in p:
            if v is None:
                return False
            try:
                if np.isnan(float(v)):
                    return False
            except (TypeError, ValueError):
                return False
    return True


# ── [핵심 수정] atan2 wraparound 보정 ─────────────────────────

def unwrap_angle_series(series: pd.Series) -> pd.Series:
    """
    atan2 결과(-180~+180)를 시계열로 언랩핑.
    NaN은 forward fill로 보간 후 언랩핑, 이후 원래 NaN 위치 복원.
    """
    was_nan = series.isna()
    filled  = series.ffill().bfill()   # NaN 임시 채움

    if filled.isna().all():
        return series

    unwrapped = np.degrees(np.unwrap(np.radians(filled.values)))
    result    = pd.Series(unwrapped, index=series.index)
    result[was_nan] = np.nan           # 원래 NaN 복원
    return result


# ── 프레임별 키네마틱 계산 ─────────────────────────────────────

def calc_kinematics(
    df: pd.DataFrame,
    throw_hand: str = "R",
    release_frame: int = None,
) -> pd.DataFrame:
    """
    전처리된 포즈 df → 프레임별 각도/플래그 DataFrame.

    [수정] 어깨/골반 기울기: rows 루프에서 raw atan2 계산 →
          DataFrame 생성 후 unwrap_angle_series() 일괄 적용
          → early_open 플래그도 unwrap 후 각도로 재계산
    release_frame: 릴리즈 프레임 번호 (지정 시 팔꿈치 유효구간 제한 적용)
    """
    if throw_hand == "R":
        t_sh, t_el, t_wr = "right_shoulder", "right_elbow", "right_wrist"
        s_hp, s_kn, s_an = "left_hip",       "left_knee",   "left_ankle"
    else:
        t_sh, t_el, t_wr = "left_shoulder",  "left_elbow",  "left_wrist"
        s_hp, s_kn, s_an = "right_hip",      "right_knee",  "right_ankle"

    rows     = []
    valid_df = df[df["detected"].fillna(False)].copy()

    for _, row in valid_df.iterrows():
        l_sh = pt(row, "left_shoulder")
        r_sh = pt(row, "right_shoulder")
        l_hp = pt(row, "left_hip")
        r_hp = pt(row, "right_hip")
        t_shoulder = pt(row, t_sh)
        t_elbow    = pt(row, t_el)
        t_wrist    = pt(row, t_wr)
        s_hip      = pt(row, s_hp)
        s_knee     = pt(row, s_kn)
        s_ankle    = pt(row, s_an)

        # raw atan2 계산 (unwrap은 아래서 일괄 적용)
        sh_angle = (np.degrees(np.arctan2(r_sh[1]-l_sh[1], r_sh[0]-l_sh[0]))
                    if is_valid(l_sh, r_sh) else np.nan)
        hp_angle = (np.degrees(np.arctan2(r_hp[1]-l_hp[1], r_hp[0]-l_hp[0]))
                    if is_valid(l_hp, r_hp) else np.nan)

        sh_cx = (l_sh[0]+r_sh[0])/2 if is_valid(l_sh, r_sh) else np.nan
        hp_cx = (l_hp[0]+r_hp[0])/2 if is_valid(l_hp, r_hp) else np.nan

        elbow_angle = calc_angle(t_shoulder, t_elbow, t_wrist)
        knee_angle  = calc_angle(s_hip, s_knee, s_ankle)
        hp_height_diff = abs(l_hp[1]-r_hp[1]) if is_valid(l_hp, r_hp) else np.nan

        # [수정] sh-hp 차이: raw atan2로 wrapped difference 계산 (독립 unwrap 금지)
        if not np.isnan(sh_angle) and not np.isnan(hp_angle):
            raw_diff    = sh_angle - hp_angle
            sh_hp_diff  = ((raw_diff + 180) % 360) - 180   # [-180, 180] normalize
        else:
            sh_hp_diff  = np.nan

        rows.append({
            "frame":           row["frame"],
            "time_sec":        row["time_sec"],
            "shoulder_angle":  sh_angle,
            "hip_angle":       hp_angle,
            "sh_hp_diff":      sh_hp_diff,
            "elbow_angle":     elbow_angle,
            "knee_angle":      knee_angle,
            "hip_height_diff": hp_height_diff,
            "sh_cx":           sh_cx,
            "hp_cx":           hp_cx,
            "t_wrist_x":       t_wrist[0] if is_valid(t_wrist) else np.nan,  # 키네마틱 시퀀스용
            "t_wrist_y":       t_wrist[1] if is_valid(t_wrist) else np.nan,
            "t_elbow_y":       t_elbow[1] if is_valid(t_elbow) else np.nan,
        })

    if not rows:
        return pd.DataFrame()

    kin = pd.DataFrame(rows)

    # ── unwrap: 시각화용 시계열 연속성 보정 (shoulder/hip 각각 독립) ──
    kin["shoulder_angle"] = unwrap_angle_series(kin["shoulder_angle"])
    kin["hip_angle"]      = unwrap_angle_series(kin["hip_angle"])

    # ── [수정] early_open: wrapped diff 기반으로 계산 ───────────────
    kin["early_open"] = np.where(
        kin["sh_hp_diff"].isna(),
        np.nan,
        (kin["sh_hp_diff"] > 15).astype(float)
    )

    # ── [수정] 팔꿈치 각도: 유효 구간 외 NaN 처리 ───────────────
    if release_frame is not None:
        invalid_elbow = (
            (kin["frame"] < release_frame - ELBOW_VALID_PRE) |
            (kin["frame"] > release_frame + ELBOW_VALID_POST)
        )
        kin.loc[invalid_elbow, "elbow_angle"] = np.nan

    # ── 파생 플래그 재계산 (unwrap 후) ──────────────────────────
    kin["wrist_below_elbow"] = np.where(
        kin["t_wrist_y"].isna() | kin["t_elbow_y"].isna(),
        np.nan,
        (kin["t_wrist_y"] > kin["t_elbow_y"]).astype(float)
    )
    kin["trunk_ahead_of_hip"] = np.where(
        kin["sh_cx"].isna() | kin["hp_cx"].isna(),
        np.nan,
        (kin["sh_cx"] < kin["hp_cx"] - 0.05).astype(float)
    )
    kin["knee_collapse"] = np.where(
        kin["knee_angle"].isna(),
        np.nan,
        (kin["knee_angle"] < 150).astype(float)
    )
    kin["arm_flyout"] = np.where(
        kin["elbow_angle"].isna(),
        np.nan,
        (kin["elbow_angle"] > 160).astype(float)
    )

    # 내부용 컬럼 제거 (wrist_x/y는 키네마틱 시퀀스 계산용으로 유지)
    kin = kin.drop(columns=["sh_cx", "hp_cx", "t_elbow_y"], errors="ignore")

    return kin


# ── 영상별 요약 집계 ───────────────────────────────────────────

def summarize(kin_df: pd.DataFrame, play_id: str, release_frame: int = None) -> dict:
    if kin_df.empty:
        return {"play_id": play_id}

    d = {"play_id": play_id, "n_frames": len(kin_df)}

    for col in ["shoulder_angle", "hip_angle", "elbow_angle",
                "knee_angle", "hip_height_diff"]:
        s = kin_df[col].dropna()
        d[f"{col}_mean"] = round(s.mean(), 2) if len(s) else np.nan
        d[f"{col}_max"]  = round(s.max(),  2) if len(s) else np.nan
        d[f"{col}_min"]  = round(s.min(),  2) if len(s) else np.nan
        d[f"{col}_std"]  = round(s.std(),  2) if len(s) else np.nan  # [추가] 표준편차

    # 플래그 rate: 팔로우스루 오염 방지 → release_frame 이전 구간만 사용
    if "frame" in kin_df.columns and release_frame is not None:
        flag_df = kin_df[kin_df["frame"] <= release_frame]
    else:
        flag_df = kin_df

    for flag in ["early_open", "wrist_below_elbow", "trunk_ahead_of_hip",
                 "knee_collapse", "arm_flyout"]:
        s = flag_df[flag].dropna()
        d[f"{flag}_rate"] = round(s.mean(), 3) if len(s) else np.nan

    # sh_hp_diff: wrapped difference [-180, 180] 기반, pre-release 구간만
    if "sh_hp_diff" in flag_df.columns:
        diff = flag_df["sh_hp_diff"].dropna()
    else:
        diff = pd.Series(dtype=float)
    d["sh_hp_diff_mean"] = round(diff.mean(), 2) if len(diff) else np.nan
    d["sh_hp_diff_std"]  = round(diff.std(),  2) if len(diff) else np.nan

    # 릴리즈 직전 5프레임 평균 (가장 중요한 구간)
    if "frame" in kin_df.columns and release_frame is not None:
        pre5 = kin_df[
            (kin_df["frame"] >= release_frame - 5) &
            (kin_df["frame"] <= release_frame)
        ]
        for col in ["shoulder_angle", "hip_angle", "knee_angle"]:
            s = pre5[col].dropna()
            d[f"{col}_pre5_mean"] = round(s.mean(), 2) if len(s) else np.nan

        # 팔꿈치 pre5: 물리적으로 불가능한 값(< 30°) 제외
        elb = pre5["elbow_angle"].dropna()
        elb = elb[elb >= 30]   # 30° 미만 = 포즈 오검출로 간주
        d["elbow_angle_pre5_mean"] = round(elb.mean(), 2) if len(elb) else np.nan

    return d


# ── 키네마틱 시퀀스 ────────────────────────────────────────────

def calc_kinematic_sequence(
    kin_df: pd.DataFrame,
    release_frame: int,
    smooth_win: int = 5,
    search_pre: int = 25,
) -> tuple:
    """
    골반 → 몸통 → 팔꿈치 → 손목 각속도 피크 타이밍 분석.

    각속도:
      - 골반(pelvis): d(hip_angle)/dt
      - 몸통(trunk) : d(shoulder_angle)/dt
      - 팔꿈치(elbow): d(elbow_angle)/dt  (신전 속도)
      - 손목(wrist)  : sqrt(dx² + dy²) / dt  (선속도)

    피크는 절댓값 기준 (회전 방향 무관).
    탐색 구간: release_frame - search_pre ~ release_frame

    Returns:
        summary (dict): 피크 프레임 오프셋, 속도, 순서 여부, 타이밍 갭
        vel_df (DataFrame): 프레임별 각속도 시계열 (시각화용)
    """
    pre = kin_df[
        (kin_df["frame"] >= release_frame - search_pre) &
        (kin_df["frame"] <= release_frame)
    ].copy()

    if len(pre) < 5:
        return {}, pd.DataFrame()

    frames = pre["frame"].values

    def _vel(series):
        """NaN ffill → gradient → 스무딩 → 절댓값"""
        arr = series.ffill().bfill().values.astype(float)
        if np.isnan(arr).all():
            return np.full(len(arr), np.nan)
        v = np.gradient(arr)
        v = uniform_filter1d(v, size=smooth_win)
        return np.abs(v)

    hip_vel    = _vel(pre["hip_angle"])
    trunk_vel  = _vel(pre["shoulder_angle"])
    elbow_vel  = _vel(pre["elbow_angle"]) if "elbow_angle" in pre else np.full(len(pre), np.nan)

    # 손목 선속도
    if "t_wrist_x" in pre.columns and "t_wrist_y" in pre.columns:
        wx = pre["t_wrist_x"].ffill().bfill().values.astype(float)
        wy = pre["t_wrist_y"].ffill().bfill().values.astype(float)
        wrist_vel = np.sqrt(np.gradient(wx)**2 + np.gradient(wy)**2)
        wrist_vel = uniform_filter1d(wrist_vel, size=smooth_win)
    else:
        wrist_vel = np.full(len(pre), np.nan)

    vel_df = pd.DataFrame({
        "frame":     frames,
        "hip_vel":   hip_vel,
        "trunk_vel": trunk_vel,
        "elbow_vel": elbow_vel,
        "wrist_vel": wrist_vel,
    })

    def _peak_offset(vel_arr):
        """피크 인덱스 → release 기준 프레임 오프셋"""
        if np.isnan(vel_arr).all():
            return np.nan, np.nan
        idx = int(np.nanargmax(vel_arr))
        offset = int(frames[idx]) - release_frame   # 음수 = 릴리즈 전
        return offset, float(vel_arr[idx])

    hip_off,   hip_peak   = _peak_offset(hip_vel)
    trunk_off, trunk_peak = _peak_offset(trunk_vel)
    elbow_off, elbow_peak = _peak_offset(elbow_vel)
    wrist_off, wrist_peak = _peak_offset(wrist_vel)

    # 순서 검증: 골반 → 몸통 → 팔꿈치 → 손목 (오프셋이 증가해야 함)
    offsets = [o for o in [hip_off, trunk_off, elbow_off, wrist_off]
               if not (isinstance(o, float) and np.isnan(o))]
    seq_correct = all(offsets[i] <= offsets[i+1] for i in range(len(offsets)-1))

    def _gap(a, b):
        if np.isnan(a) or np.isnan(b):
            return np.nan
        return round(float(b - a), 1)

    summary = {
        "seq_hip_offset":   hip_off,    # 릴리즈 기준 프레임 오프셋 (음수=앞)
        "seq_trunk_offset": trunk_off,
        "seq_elbow_offset": elbow_off,
        "seq_wrist_offset": wrist_off,
        "seq_hip_peak_vel":   round(hip_peak,   3) if not np.isnan(hip_peak)   else np.nan,
        "seq_trunk_peak_vel": round(trunk_peak, 3) if not np.isnan(trunk_peak) else np.nan,
        "seq_elbow_peak_vel": round(elbow_peak, 3) if not np.isnan(elbow_peak) else np.nan,
        "seq_wrist_peak_vel": round(wrist_peak, 3) if not np.isnan(wrist_peak) else np.nan,
        "seq_correct":        seq_correct,
        "seq_hip_to_trunk":   _gap(hip_off, trunk_off),    # 프레임 수
        "seq_trunk_to_elbow": _gap(trunk_off, elbow_off),
        "seq_elbow_to_wrist": _gap(elbow_off, wrist_off),
    }
    return summary, vel_df


# ── 메인 파이프라인 ────────────────────────────────────────────

def run_pipeline():
    proc_files = sorted(POSE_DIR.glob("*_trimmed_yolo_pose_proc.csv"))
    print(f"전처리 파일: {len(proc_files)}개\n")

    meta = pd.read_csv(META_CSV) if META_CSV.exists() else pd.DataFrame()

    summaries = []
    for i, fpath in enumerate(proc_files):
        play_id = fpath.stem.replace("_trimmed_yolo_pose_proc", "")
        print(f"[{i+1}/{len(proc_files)}] {play_id[:8]}...")

        df = pd.read_csv(fpath)

        from pose_preprocessor import XY_COLS
        valid_cols = [c for c in XY_COLS if c in df.columns]
        nan_ratio  = df[valid_cols].isna().mean().mean()
        if nan_ratio > NAN_THRESH:
            print(f"  ✗ 제외 (NaN {nan_ratio:.0%})")
            continue

        rel_frame, rel_time = detect_release(df, throw_hand=THROW_HAND, method="velocity")
        if rel_frame is None:
            rel_frame, rel_time = detect_release(df, throw_hand=THROW_HAND, method="x_min")
        if rel_frame is None:
            print(f"  ✗ 릴리즈 감지 실패")
            continue

        df_win = extract_window(df, rel_frame, pre=PRE_FRAMES, post=POST_FRAMES)
        n_valid = df_win["detected"].fillna(False).sum()
        if n_valid < MIN_FRAMES:
            print(f"  ✗ 윈도우 유효 프레임 부족 ({n_valid})")
            continue

        print(f"  릴리즈 frame={rel_frame} ({rel_time:.2f}s) | 윈도우 {len(df_win)}f")
        kin_df = calc_kinematics(df_win, throw_hand=THROW_HAND, release_frame=rel_frame)

        kin_path = POSE_DIR / f"{play_id}_kinematics.csv"
        kin_df.to_csv(kin_path, index=False)

        summary = summarize(kin_df, play_id, release_frame=rel_frame)

        # 키네마틱 시퀀스
        seq_summary, vel_df = calc_kinematic_sequence(kin_df, release_frame=rel_frame)
        summary.update(seq_summary)
        if not vel_df.empty:
            vel_df.to_csv(POSE_DIR / f"{play_id}_kinseq.csv", index=False)

        summaries.append(summary)

        eo  = summary.get("early_open_rate", float("nan"))
        elb = summary.get("elbow_angle_mean", float("nan"))
        seq = "✓" if seq_summary.get("seq_correct") else "✗"
        print(f"  ✓ {len(kin_df)}f | early_open {eo:.0%} | elbow {elb:.1f}° | seq {seq}")

    if not summaries:
        print("분석된 데이터 없음")
        return

    result = pd.DataFrame(summaries)

    if not meta.empty:
        merge_cols = [c for c in ["play_id", "game_date", "pitch_name",
                                   "release_speed", "event", "balls", "strikes"]
                      if c in meta.columns]
        result = result.merge(meta[merge_cols], on="play_id", how="left")

    result.to_csv(OUT_CSV, index=False)
    print(f"\n저장: {OUT_CSV}")
    print(f"분석 완료: {len(result)}개 투구\n")

    view_cols = [c for c in [
        "play_id", "game_date", "release_speed",
        "elbow_angle_pre5_mean", "knee_angle_pre5_mean",
        "early_open_rate", "sh_hp_diff_mean", "sh_hp_diff_std",
    ] if c in result.columns]
    print(result[view_cols].to_string(index=False))


if __name__ == "__main__":
    run_pipeline()