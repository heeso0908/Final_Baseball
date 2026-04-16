"""
Leiter 볼넷 타석 영상 다운로드 + 포즈 비교 분석
=================================================
leiter_walks_meta.csv에 저장된 67개 볼넷 play_id로
sporty-videos 영상을 다운로드 후 삼진 타석과 포즈 비교

실행:
    python download_walks.py --max 20          # 볼넷 최대 20개 다운
    python download_walks.py --dry-run         # URL 확인만
    python download_walks.py --analyze-only    # 다운 건너뛰고 비교 분석만
"""

import os
import sys
import time
import argparse
import subprocess
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ── 경로 설정 ──
BASE_DIR   = Path(__file__).parent
WALKS_CSV  = BASE_DIR / "pose_output/leiter/leiter_walks_meta.csv"
VIDEO_DIR  = BASE_DIR / "videos/leiter"
POSE_DIR   = BASE_DIR / "pose_output/leiter"
EXISTING_META = BASE_DIR / "collection_output/leiter_videos.csv"


# ──────────────────────────────────────────
# 1. 다운로드 유틸
# ──────────────────────────────────────────

def _get_ytdlp_cmd() -> list[str] | None:
    """yt-dlp 실행 커맨드 반환 (모듈 방식 우선)"""
    py = sys.executable
    try:
        ret = subprocess.run([py, "-m", "yt_dlp", "--version"],
                             capture_output=True, check=True, timeout=5)
        return [py, "-m", "yt_dlp"]
    except Exception:
        pass
    try:
        ret = subprocess.run(["yt-dlp", "--version"],
                             capture_output=True, check=True, timeout=5)
        return ["yt-dlp"]
    except Exception:
        pass
    return None


def download_video(play_id: str, save_path: str, dry_run: bool = False) -> str:
    """
    반환: "ok" | "skip" | "fail" | "dry_run"
    """
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10_000:
        print(f"  → skip (이미 존재): {Path(save_path).name}")
        return "skip"

    sporty_url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"

    if dry_run:
        print(f"  [dry-run] {sporty_url}")
        return "dry_run"

    ytdlp_cmd = _get_ytdlp_cmd()
    if ytdlp_cmd:
        ret = subprocess.run(
            ytdlp_cmd + [sporty_url, "-o", save_path,
                         "--quiet", "--no-warnings", "--merge-output-format", "mp4"],
            capture_output=True, timeout=90,
        )
        if ret.returncode == 0 and os.path.exists(save_path):
            mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"  ✓ {Path(save_path).name} ({mb:.1f} MB)")
            return "ok"
        err = ret.stderr.decode(errors="ignore")[:100]
        print(f"  ✗ yt-dlp 실패: {err}")

    # Direct API fallback
    api_url = f"https://baseballsavant.mlb.com/api/video/search?playId={play_id}"
    try:
        res = requests.get(api_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            for item in res.json():
                if isinstance(item, dict) and ".mp4" in item.get("url", ""):
                    mp4 = requests.get(item["url"], stream=True, timeout=60,
                                       headers={"User-Agent": "Mozilla/5.0"})
                    if mp4.status_code == 200:
                        os.makedirs(os.path.dirname(save_path), exist_ok=True)
                        with open(save_path, "wb") as f:
                            for chunk in mp4.iter_content(65536):
                                f.write(chunk)
                        mb = os.path.getsize(save_path) / 1024 / 1024
                        print(f"  ✓ (direct) {Path(save_path).name} ({mb:.1f} MB)")
                        return "ok"
    except Exception as e:
        print(f"  ✗ direct 실패: {e}")

    print(f"  ⚠ 실패 — URL: {sporty_url}")
    return "fail"


# ──────────────────────────────────────────
# 2. 볼넷 영상 수집
# ──────────────────────────────────────────

def collect_walk_videos(max_videos: int = 20, dry_run: bool = False) -> pd.DataFrame:
    """
    leiter_walks_meta.csv → 볼넷 영상 다운로드
    - 3-2 카운트 우선 (최대 압박 상황)
    - 이미 존재하는 play_id 건너뜀
    """
    walks = pd.read_csv(WALKS_CSV)
    print(f"\n볼넷 타석 전체: {len(walks)}개")

    # 기존 다운로드된 play_id 확인
    existing_ids = set()
    if EXISTING_META.exists():
        ex = pd.read_csv(EXISTING_META)
        existing_ids = set(ex[ex["download_status"].isin(["ok","skip"])]["play_id"].dropna())

    # 이미 있는 영상 파일도 확인
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    existing_files = {p.stem for p in VIDEO_DIR.glob("*.mp4")}
    existing_ids |= existing_files

    # 3-2 → 3-1 → 3-0 순 정렬 (스트라이크 많을수록 압박 큼)
    walks["count_priority"] = walks["strikes"].map({2: 0, 1: 1, 0: 2})
    walks = walks.sort_values(["count_priority", "release_speed"], ascending=[True, False])

    print(f"카운트 분포: {walks['strikes'].value_counts().to_dict()}")
    print(f"기존 다운로드 제외: {len(existing_ids)}개")

    results = []
    downloaded = 0

    for _, row in walks.iterrows():
        if downloaded >= max_videos:
            break

        play_id = str(row["play_id"])
        if play_id in existing_ids:
            print(f"  → skip: {play_id[:8]}... (기존 존재)")
            continue

        save_path = str(VIDEO_DIR / f"{play_id}.mp4")
        count_str = f"{int(row['balls'])}-{int(row['strikes'])}"
        print(f"\n[{downloaded+1}/{max_videos}] {row['game_date']} | 볼넷 | {count_str} | "
              f"{row.get('pitch_name','?')} {row.get('release_speed','?')}mph")

        status = download_video(play_id, save_path, dry_run=dry_run)

        results.append({
            "play_id":       play_id,
            "game_date":     row["game_date"],
            "balls":         row["balls"],
            "strikes":       row["strikes"],
            "pitch_name":    row.get("pitch_name"),
            "release_speed": row.get("release_speed"),
            "release_pos_x": row.get("release_pos_x"),
            "release_pos_z": row.get("release_pos_z"),
            "event":         "walk",
            "download_status": status,
            "video_path":    save_path if status in ("ok","skip") else None,
        })

        if status in ("ok", "skip", "dry_run"):
            downloaded += 1

        if not dry_run:
            time.sleep(1.5)

    df = pd.DataFrame(results)
    out_path = POSE_DIR / "leiter_walks_downloaded.csv"
    df.to_csv(out_path, index=False)
    print(f"\n✓ 메타데이터 저장: {out_path}")
    print(f"  성공: {(df['download_status']=='ok').sum()} | 기존: {(df['download_status']=='skip').sum()} | 실패: {(df['download_status']=='fail').sum()}")
    return df


# ──────────────────────────────────────────
# 3. 포즈 추출 (볼넷 영상)
# ──────────────────────────────────────────

def run_pose_extraction(walk_meta: pd.DataFrame) -> list[dict]:
    """다운로드 완료된 볼넷 영상 포즈 추출"""
    from pose_extractor import extract_pose, analyze_pitching

    ok_videos = walk_meta[walk_meta["download_status"].isin(["ok", "skip"])]
    ok_videos = ok_videos[ok_videos["video_path"].notna()]
    print(f"\n포즈 추출 대상: {len(ok_videos)}개 볼넷 영상")

    records = []
    for _, row in ok_videos.iterrows():
        vpath = row["video_path"]
        if not os.path.exists(vpath):
            print(f"  ✗ 파일 없음: {vpath}")
            continue

        play_id = row["play_id"]
        print(f"\n  {play_id[:8]}... ({row['game_date']} | {row['balls']}-{row['strikes']})")

        try:
            csv_out = str(POSE_DIR / f"walks/{play_id}_pose.csv")
            os.makedirs(os.path.dirname(csv_out), exist_ok=True)

            if os.path.exists(csv_out):
                pose_df = pd.read_csv(csv_out)
                print(f"  → 기존 포즈 CSV 로드")
            else:
                pose_df = extract_pose(vpath, output_dir=str(POSE_DIR / "walks"))

            if pose_df["detected"].sum() < 5:
                print(f"  ✗ 포즈 감지 부족 ({pose_df['detected'].sum()}프레임)")
                continue

            fps = 30.0
            pitch_df = analyze_pitching(pose_df, throw_hand="R")
            n = len(pitch_df)

            records.append({
                "play_id":           play_id,
                "event":             "walk",
                "balls":             row["balls"],
                "strikes":           row["strikes"],
                "pitch_name":        row.get("pitch_name"),
                "release_speed":     row.get("release_speed"),
                "release_pos_x":     row.get("release_pos_x"),
                "release_pos_z":     row.get("release_pos_z"),
                # 포즈 지표
                "early_open_rate":   pitch_df["early_open"].mean(),
                "knee_collapse_rate": pitch_df["knee_collapse"].mean(),
                "arm_flyout_rate":   pitch_df["arm_flyout"].mean(),
                "trunk_ahead_rate":  pitch_df["trunk_ahead_of_hip"].mean(),
                "wrist_below_elbow_rate": pitch_df["wrist_below_elbow"].mean(),
                "shoulder_angle_mean": pitch_df["shoulder_angle"].mean(),
                "hip_angle_mean":    pitch_df["hip_angle"].mean(),
                "front_knee_mean":   pitch_df["front_knee_angle"].mean(),
                "throw_elbow_mean":  pitch_df["throw_elbow_angle"].mean(),
                "n_frames":          n,
            })
            print(f"  ✓ early_open={pitch_df['early_open'].mean():.1%} | "
                  f"knee_collapse={pitch_df['knee_collapse'].mean():.1%}")

        except Exception as e:
            print(f"  ✗ 오류: {e}")

    return records


# ──────────────────────────────────────────
# 4. 삼진 vs 볼넷 비교 분석
# ──────────────────────────────────────────

def compare_strikeout_vs_walk(
    walk_records: list[dict],
    so_meta_path: str = str(EXISTING_META),
) -> pd.DataFrame:
    """삼진 vs 볼넷 포즈 지표 비교"""
    # 삼진 포즈 기존 분석 결과 로드
    pitching_csv = POSE_DIR / "leiter_pitching_analysis.csv"
    if not pitching_csv.exists():
        print(f"삼진 분석 CSV 없음: {pitching_csv}")
        return pd.DataFrame()

    so_df = pd.read_csv(pitching_csv)

    # 볼넷 집계
    if not walk_records:
        print("볼넷 포즈 데이터 없음")
        return pd.DataFrame()

    walk_df = pd.DataFrame(walk_records)

    metrics = [
        "early_open_rate", "knee_collapse_rate", "arm_flyout_rate",
        "trunk_ahead_rate", "wrist_below_elbow_rate",
    ]

    print("\n" + "=" * 55)
    print("Leiter 삼진 vs 볼넷 — 포즈 지표 비교")
    print("=" * 55)
    print(f"{'지표':<28} {'삼진':>8} {'볼넷':>8} {'차이':>8}")
    print("-" * 55)

    # 삼진 포즈 지표는 pitching_analysis.csv에서 집계
    so_metrics = {
        "early_open_rate":        so_df["early_open"].mean() if "early_open" in so_df else np.nan,
        "knee_collapse_rate":     so_df["knee_collapse"].mean() if "knee_collapse" in so_df else np.nan,
        "arm_flyout_rate":        so_df["arm_flyout"].mean() if "arm_flyout" in so_df else np.nan,
        "trunk_ahead_rate":       so_df["trunk_ahead_of_hip"].mean() if "trunk_ahead_of_hip" in so_df else np.nan,
        "wrist_below_elbow_rate": so_df["wrist_below_elbow"].mean() if "wrist_below_elbow" in so_df else np.nan,
    }

    rows = []
    for m in metrics:
        so_val  = so_metrics.get(m, np.nan)
        wk_val  = walk_df[m].mean() if m in walk_df.columns else np.nan
        diff    = wk_val - so_val if not (np.isnan(so_val) or np.isnan(wk_val)) else np.nan
        label   = m.replace("_rate", "").replace("_", " ")
        print(f"  {label:<26} {so_val:>7.1%}  {wk_val:>7.1%}  {diff:>+7.1%}")
        rows.append({"metric": m, "strikeout": so_val, "walk": wk_val, "diff": diff})

    print()
    print("릴리즈 포인트 (Statcast 실측)")
    print(f"  {'release_pos_x (볼넷)':<26} {walk_df['release_pos_x'].mean():>7.3f}")
    print(f"  {'release_pos_z (볼넷)':<26} {walk_df['release_pos_z'].mean():>7.3f}")

    result_df = pd.DataFrame(rows)
    out = POSE_DIR / "leiter_so_vs_walk_comparison.csv"
    result_df.to_csv(out, index=False)
    print(f"\n✓ 비교 결과 저장: {out}")
    return result_df


# ──────────────────────────────────────────
# 5. 실행
# ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",          type=int,  default=20,    help="다운로드 최대 볼넷 영상 수")
    parser.add_argument("--dry-run",      action="store_true",      help="URL 확인만 (다운로드 안 함)")
    parser.add_argument("--analyze-only", action="store_true",      help="다운로드 건너뛰고 포즈 분석만")
    parser.add_argument("--no-pose",      action="store_true",      help="포즈 추출 건너뜀")
    args = parser.parse_args()

    print("=" * 60)
    print("Leiter 볼넷 타석 영상 수집 + 포즈 비교 분석")
    print("=" * 60)

    walk_records = []

    if not args.analyze_only:
        walk_meta = collect_walk_videos(
            max_videos=args.max,
            dry_run=args.dry_run,
        )
    else:
        # 기존 메타 로드
        dl_csv = POSE_DIR / "leiter_walks_downloaded.csv"
        if dl_csv.exists():
            walk_meta = pd.read_csv(dl_csv)
            print(f"기존 다운로드 메타 로드: {len(walk_meta)}개")
        else:
            print("다운로드 메타 없음. --analyze-only 전에 다운로드 먼저 실행 필요.")
            sys.exit(1)

    if not args.dry_run and not args.no_pose:
        walk_records = run_pose_extraction(walk_meta)
        if walk_records:
            compare_strikeout_vs_walk(walk_records)
    elif args.dry_run:
        print("\n[dry-run] 다운로드 없이 URL만 확인했습니다.")

    print("\n완료!")
