"""
Leiter 삼진 타석 영상 다운로드
================================
pybaseball로 2025 Leiter 삼진 play_id 수집 → sporty-videos 영상 다운로드

실행:
    python download_strikeouts.py --max 20       # 최대 20개
    python download_strikeouts.py --dry-run      # URL 확인만
    python download_strikeouts.py --fetch-only   # 메타만 저장, 다운 안 함
"""

import os
import sys
import time
import argparse
import subprocess
import pandas as pd
from pathlib import Path

BASE_DIR  = Path(__file__).parent
SO_META   = BASE_DIR / "pose_output/leiter/leiter_so_meta.csv"
SO_DL_CSV = BASE_DIR / "pose_output/leiter/leiter_so_downloaded.csv"
VIDEO_DIR = BASE_DIR / "videos/leiter_so"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)


# ── yt-dlp 커맨드 ───────────────────────────────────────────────

def _get_ytdlp_cmd():
    py = sys.executable
    for cmd in [[py, "-m", "yt_dlp"], ["yt-dlp"]]:
        try:
            subprocess.run(cmd + ["--version"], capture_output=True, check=True, timeout=5)
            return cmd
        except Exception:
            continue
    return None


def download_video(play_id: str, save_path: str, dry_run: bool = False) -> str:
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10_000:
        return "skip"

    url = f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"
    if dry_run:
        print(f"  [dry-run] {url}")
        return "dry_run"

    cmd = _get_ytdlp_cmd()
    if not cmd:
        print("  ✗ yt-dlp 없음")
        return "fail"

    ret = subprocess.run(
        cmd + [url, "-o", save_path,
               "--quiet", "--no-warnings", "--merge-output-format", "mp4"],
        capture_output=True, timeout=90,
    )
    if ret.returncode == 0 and os.path.exists(save_path):
        mb = os.path.getsize(save_path) / 1024 / 1024
        print(f"  ✓ {Path(save_path).name} ({mb:.1f} MB)")
        return "ok"
    err = ret.stderr.decode(errors="ignore")[:120]
    print(f"  ✗ 실패: {err}")
    return "fail"


# ── Statcast 삼진 데이터 수집 ────────────────────────────────────

def fetch_so_meta(force: bool = False) -> pd.DataFrame:
    if SO_META.exists() and not force:
        df = pd.read_csv(SO_META)
        print(f"기존 메타 로드: {len(df)}개")
        return df

    print("pybaseball로 Leiter 2025 데이터 수집 중...")
    try:
        from pybaseball import statcast_pitcher
        import requests
    except ImportError:
        print("  pip install pybaseball requests 필요")
        sys.exit(1)

    # Jack Leiter MLB ID: 694973
    df = statcast_pitcher("2025-03-01", "2025-11-01", player_id=694973)

    # 삼진: events = 'strikeout' 또는 'strikeout_double_play'
    so_events = {"strikeout", "strikeout_double_play"}
    so = df[df["events"].isin(so_events)].copy()
    print(f"  삼진 타석 수: {len(so)}개")

    # play_id 수집: sv_id 우선, fallback은 MLB Stats API
    play_cache = {}

    def resolve_play_id(row):
        sv = str(row.get("sv_id", ""))
        if sv not in ("", "None", "nan", "<NA>", "NaT"):
            return sv
        try:
            game_pk = int(row["game_pk"])
            at_bat  = int(row["at_bat_number"])
            if game_pk not in play_cache:
                url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
                res = requests.get(url, timeout=10)
                play_cache[game_pk] = res.json().get("allPlays", []) if res.status_code == 200 else []
            plays = play_cache[game_pk]
            play  = next((p for p in plays if p.get("atBatIndex") == at_bat - 1), None)
            if play:
                pid = play.get("playEvents", [{}])[-1].get("playId")
                if pid:
                    return pid
        except Exception:
            pass
        return None

    print("  play_id 수집 중 (MLB Stats API)...")
    so = so.copy()
    so["play_id"] = so.apply(resolve_play_id, axis=1)
    so = so.dropna(subset=["play_id"])
    print(f"  play_id 확보: {len(so)}개")

    cols = ["play_id", "game_date", "balls", "strikes",
            "pitch_name", "release_pos_x", "release_pos_z", "release_speed", "events"]
    so = so[[c for c in cols if c in so.columns]]

    # 정규시즌만 (4월 이후) — 스프링 트레이닝 제외
    so["game_date"] = pd.to_datetime(so["game_date"])
    so = so[so["game_date"] >= "2025-04-01"].copy()
    so["game_date"] = so["game_date"].astype(str)
    print(f"  정규시즌 필터 후: {len(so)}개")

    # 구속 높은 순 (4심, 싱커 우선)
    so = so.sort_values("release_speed", ascending=False).reset_index(drop=True)

    SO_META.parent.mkdir(parents=True, exist_ok=True)
    so.to_csv(SO_META, index=False)
    print(f"  저장: {SO_META}")
    return so


# ── 메인 ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max",        type=int, default=20)
    parser.add_argument("--dry-run",    action="store_true")
    parser.add_argument("--fetch-only", action="store_true", help="메타만 저장")
    parser.add_argument("--force-fetch",action="store_true", help="메타 재수집")
    args = parser.parse_args()

    so_meta = fetch_so_meta(force=args.force_fetch)
    print(f"\n삼진 타석 전체: {len(so_meta)}개")

    if args.fetch_only:
        print("--fetch-only: 다운로드 스킵")
        return

    # 이미 다운된 것 제외
    existing = set()
    if SO_DL_CSV.exists():
        prev = pd.read_csv(SO_DL_CSV)
        existing = set(prev[prev["download_status"].isin(["ok","skip"])]["play_id"])
    targets = so_meta[~so_meta["play_id"].isin(existing)].head(args.max)
    print(f"기존 다운로드 제외: {len(existing)}개")
    print(f"다운로드 대상: {len(targets)}개\n")

    records = []
    ok, skip, fail = 0, 0, 0

    for i, row in targets.iterrows():
        pid   = row["play_id"]
        label = f"{row.get('game_date','?')} | {row.get('events','?')} | " \
                f"{row.get('balls',0)}-{row.get('strikes',0)} | " \
                f"{row.get('pitch_name','?')} {row.get('release_speed','?')}mph"
        print(f"[{ok+skip+fail+1}/{len(targets)}] {label}")

        save_path = str(VIDEO_DIR / f"{pid}.mp4")
        status = download_video(pid, save_path, dry_run=args.dry_run)

        records.append({
            "play_id":         pid,
            "game_date":       row.get("game_date"),
            "balls":           row.get("balls"),
            "strikes":         row.get("strikes"),
            "pitch_name":      row.get("pitch_name"),
            "release_speed":   row.get("release_speed"),
            "event":           row.get("events"),
            "download_status": status,
            "video_path":      save_path if status in ("ok","skip") else "",
        })

        if status in ("ok","dry_run"): ok += 1
        elif status == "skip":         skip += 1
        else:                          fail += 1

        if status == "ok":
            time.sleep(0.5)

    result = pd.DataFrame(records)
    if SO_DL_CSV.exists():
        prev = pd.read_csv(SO_DL_CSV)
        result = pd.concat([prev, result], ignore_index=True).drop_duplicates("play_id")
    result.to_csv(SO_DL_CSV, index=False)
    print(f"\n✓ 메타 저장: {SO_DL_CSV}")
    print(f"  성공: {ok} | 스킵: {skip} | 실패: {fail}")


if __name__ == "__main__":
    main()
