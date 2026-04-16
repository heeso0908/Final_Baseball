"""
MLB 선수 영상 수집기 — 타자/투수 분석용
===========================================
분석 대상:
  타자: Langford (K%/스윙 궤적), Jung (부상 복귀 메커니즘)
  투수: Leiter (제구/릴리즈 포인트), Winn (ERA vs FIP 괴리)

수집 전략:
  1. Statcast → play_id 확보
  2. Baseball Savant sporty-videos → yt-dlp 다운로드
  3. 실패 시 직접 MP4 URL fallback
  4. 메타데이터 CSV 함께 저장

설치:
    pip install pybaseball requests yt-dlp

실행:
    python collect_videos.py                     # 전체 수집
    python collect_videos.py --player langford   # 특정 선수만
    python collect_videos.py --dry-run           # URL만 확인 (다운로드 안 함)
    python collect_videos.py --date-range 2025-04-01 2025-04-10
"""

import os
import time
import argparse
import subprocess
import requests
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

try:
    from pybaseball import statcast_batter, statcast_pitcher
    from pybaseball import playerid_lookup
except ImportError:
    print("pybaseball 설치 필요: pip install pybaseball")
    raise


# ──────────────────────────────────────────
# 분석 대상 선수 설정
# ──────────────────────────────────────────

@dataclass
class PlayerTarget:
    name: str                          # 표시용 이름
    statcast_name: str                 # Statcast DB 이름 (Last, First 형식)
    role: str                          # "batter" | "pitcher"
    analysis_focus: str                # 분석 목적
    # 수집 필터
    min_launch_speed: Optional[float] = None   # 타자: 최소 타구속도 (mph) — 컨택 품질 필터
    pitch_types: Optional[list] = None         # 투수: 특정 구종만
    events_filter: Optional[list] = None       # 특정 결과만 (strikeout, home_run 등)
    max_videos: int = 20
    priority_events: list = field(default_factory=list)  # 우선 수집할 결과

TARGETS = [
    PlayerTarget(
        name            = "Langford",
        statcast_name   = "Langford, Wyatt",
        role            = "batter",
        analysis_focus  = "K%/스윙궤적 — Whiff% 높음, Chase% 상위권",
        # 삼진 + 컨택 성공 타석 비교를 위해 필터 없이 수집
        events_filter   = ["strikeout", "strikeout_double_play",
                           "single", "double", "triple", "home_run", "field_out"],
        priority_events = ["strikeout"],   # 삼진 타석 우선
        max_videos      = 25,
    ),
    PlayerTarget(
        name            = "Jung",
        statcast_name   = "Jung, Josh",
        role            = "batter",
        analysis_focus  = "부상 복귀 후 메커니즘 보상 — EV 감소, 하체 보상 의심",
        min_launch_speed= 85.0,           # EV 분석이라 컨택 타구 위주
        events_filter   = ["single", "double", "triple", "home_run",
                           "field_out", "strikeout"],
        priority_events = ["home_run", "double"],
        max_videos      = 20,
    ),
    PlayerTarget(
        name            = "Leiter",
        statcast_name   = "Leiter, Jack",
        role            = "pitcher",
        analysis_focus  = "BB% 높음 — 릴리즈 포인트 불안정 가설",
        # 볼넷 + 삼진 타석 비교 (카운트 불리 시 릴리즈 변화)
        events_filter   = ["walk", "strikeout", "strikeout_double_play",
                           "single", "home_run"],
        priority_events = ["walk", "strikeout"],
        pitch_types     = ["FF", "SI", "CU", "CH", "SL"],  # 전 구종
        max_videos      = 25,
    ),
    PlayerTarget(
        name            = "Webb",
        statcast_name   = "Webb, Jacob",
        role            = "pitcher",
        analysis_focus  = "ERA(3.36) vs FIP(4.30) 괴리 — 약한 컨택 유도 메커니즘 검증 (Barrel% 5.9%)",
        events_filter   = ["strikeout", "strikeout_double_play",
                           "single", "double", "home_run", "field_out"],
        priority_events = ["strikeout", "field_out"],  # 약한 컨택 유도 타석 우선
        pitch_types     = ["FF", "SI", "SL", "CH", "CU"],
        max_videos      = 20,
    ),
]


# ──────────────────────────────────────────
# 설정
# ──────────────────────────────────────────

VIDEO_DIR  = "videos"
OUTPUT_DIR = "collection_output"
DEFAULT_START = "2025-03-27"   # 2025 시즌 개막
DEFAULT_END   = "2025-09-28"


# ──────────────────────────────────────────
# 1. Statcast 데이터 수집
# ──────────────────────────────────────────

def get_player_statcast(
    target: PlayerTarget,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """선수별 Statcast 수집 + 필터링 (playerid_lookup + statcast_batter/pitcher 사용)"""

    print(f"\n[Statcast] {target.name} ({target.statcast_name})")
    print(f"  기간: {start_date} ~ {end_date}")

    # 선수 ID 조회
    last, first = target.statcast_name.split(", ")
    lookup = playerid_lookup(last, first)
    if lookup.empty:
        print(f"  ✗ 선수 ID 없음 (이름 확인 필요)")
        return pd.DataFrame()

    mlbam_id = int(lookup["key_mlbam"].iloc[0])
    print(f"  MLBAM ID: {mlbam_id}")

    # 역할에 따라 적합한 함수 사용
    if target.role == "batter":
        df = statcast_batter(start_dt=start_date, end_dt=end_date, player_id=mlbam_id)
    else:
        df = statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=mlbam_id)

    if df.empty:
        print(f"  ✗ 데이터 없음")
        return pd.DataFrame()

    print(f"  전체 pitch: {len(df)}개")

    # 결과 있는 타석(마지막 pitch)만
    result_df = df[df["events"].notna()].copy()
    print(f"  결과 있는 타석: {len(result_df)}개")

    # events 필터
    if target.events_filter:
        result_df = result_df[result_df["events"].isin(target.events_filter)].copy()
        print(f"  events 필터 후: {len(result_df)}개")

    # 타구속도 필터 (타자만)
    if target.min_launch_speed and "launch_speed" in result_df.columns:
        bbe = result_df[result_df["launch_speed"] >= target.min_launch_speed].copy()
        print(f"  EV≥{target.min_launch_speed} 필터 후: {len(bbe)}개")
        # 삼진은 EV 없으므로 별도 보존
        ks = result_df[result_df["events"].str.contains("strikeout", na=False)].copy()
        result_df = pd.concat([bbe, ks]).drop_duplicates()

    # 우선순위 이벤트를 앞으로 정렬
    if target.priority_events:
        result_df["_priority"] = result_df["events"].apply(
            lambda e: 0 if e in target.priority_events else 1
        )
        result_df = result_df.sort_values(["_priority", "game_date"], ascending=[True, False])
        result_df = result_df.drop(columns=["_priority"])

    print(f"  → 수집 대상: {min(len(result_df), target.max_videos)}개 타석")
    return result_df.reset_index(drop=True)


# ──────────────────────────────────────────
# 2. play_id → video URL
# ──────────────────────────────────────────

# game_pk 별 play 데이터 캐시 (API 중복 호출 방지)
_play_cache: dict[int, list] = {}

def _fetch_plays(game_pk: int) -> list:
    """MLB Stats API에서 game_pk의 전체 play 목록 조회 (캐시)"""
    if game_pk in _play_cache:
        return _play_cache[game_pk]
    try:
        url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/playByPlay"
        res = requests.get(url, timeout=10)
        plays = res.json().get("allPlays", []) if res.status_code == 200 else []
    except Exception:
        plays = []
    _play_cache[game_pk] = plays
    return plays



def resolve_play_id(row: pd.Series) -> Optional[str]:
    """
    행에서 play UUID 추출
    1순위: sv_id (유효한 경우)
    2순위: game_pk + at_bat_number → MLB Stats API 조회
    """
    # sv_id 유효하면 그대로 사용
    sv = str(row.get("sv_id", ""))
    if sv not in ("", "None", "nan", "<NA>", "NaT"):
        return sv

    # MLB Stats API fallback
    try:
        game_pk      = int(row["game_pk"])
        at_bat_number = int(row["at_bat_number"])
        plays = _fetch_plays(game_pk)
        # atBatIndex는 0-indexed, at_bat_number는 1-indexed
        play = next((p for p in plays if p.get("atBatIndex") == at_bat_number - 1), None)
        if play:
            play_id = play.get("playEvents", [{}])[-1].get("playId")
            if play_id:
                return play_id
    except Exception:
        pass
    return None


def get_sporty_video_url(play_id: str) -> str:
    """Baseball Savant sporty-videos 페이지 URL"""
    return f"https://baseballsavant.mlb.com/sporty-videos?playId={play_id}"


def get_direct_mp4_url(play_id: str) -> Optional[str]:
    """직접 MP4 URL 시도 (Savant API)"""
    api_url = f"https://baseballsavant.mlb.com/api/video/search?playId={play_id}"
    try:
        res = requests.get(api_url, timeout=10,
                           headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            data = res.json()
            for item in data:
                if isinstance(item, dict) and ".mp4" in item.get("url", ""):
                    return item["url"]
    except Exception:
        pass
    return None


# ──────────────────────────────────────────
# 3. 다운로드
# ──────────────────────────────────────────

def _yt_dlp_available() -> bool:
    try:
        subprocess.run(["yt-dlp", "--version"],
                       capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def download_via_ytdlp(sporty_url: str, save_path: str) -> bool:
    """yt-dlp로 sporty-videos 페이지에서 MP4 추출 후 저장"""
    try:
        ret = subprocess.run(
            [
                "yt-dlp",
                sporty_url,
                "-o", save_path,
                "--quiet",
                "--no-warnings",
                "--merge-output-format", "mp4",
            ],
            capture_output=True,
            timeout=60,
        )
        if ret.returncode == 0 and os.path.exists(save_path):
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"  ✓ yt-dlp 완료: {Path(save_path).name} ({size_mb:.1f} MB)")
            return True
        else:
            err = ret.stderr.decode(errors="ignore")[:120]
            print(f"  ✗ yt-dlp 실패: {err}")
            return False
    except subprocess.TimeoutExpired:
        print("  ✗ yt-dlp 타임아웃")
        return False
    except Exception as e:
        print(f"  ✗ yt-dlp 오류: {e}")
        return False


def download_direct_mp4(url: str, save_path: str) -> bool:
    """직접 MP4 URL 다운로드"""
    try:
        res = requests.get(url, stream=True, timeout=60,
                           headers={"User-Agent": "Mozilla/5.0"})
        if res.status_code == 200:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "wb") as f:
                for chunk in res.iter_content(chunk_size=65536):
                    f.write(chunk)
            size_mb = os.path.getsize(save_path) / 1024 / 1024
            print(f"  ✓ 직접 다운로드 완료: {Path(save_path).name} ({size_mb:.1f} MB)")
            return True
        else:
            print(f"  ✗ HTTP {res.status_code}")
            return False
    except Exception as e:
        print(f"  ✗ 다운로드 실패: {e}")
        return False


def download_video(play_id: str, save_dir: str, dry_run: bool = False) -> dict:
    """
    play_id → 영상 다운로드
    반환: {"status": "ok"|"skip"|"fail"|"dry_run", "path": str|None, "url": str}
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{play_id}.mp4")

    # 이미 존재
    if os.path.exists(save_path) and os.path.getsize(save_path) > 10_000:
        print(f"  → 이미 존재: {play_id}.mp4")
        return {"status": "skip", "path": save_path, "url": ""}

    sporty_url = get_sporty_video_url(play_id)

    if dry_run:
        print(f"  [dry-run] {sporty_url}")
        return {"status": "dry_run", "path": None, "url": sporty_url}

    # 방법 1: yt-dlp
    if _yt_dlp_available():
        if download_via_ytdlp(sporty_url, save_path):
            return {"status": "ok", "path": save_path, "url": sporty_url}

    # 방법 2: 직접 MP4 URL
    direct_url = get_direct_mp4_url(play_id)
    if direct_url:
        if download_direct_mp4(direct_url, save_path):
            return {"status": "ok", "path": save_path, "url": direct_url}

    # 방법 3: URL만 반환 (수동 다운로드용)
    print(f"  ⚠ 자동 다운로드 실패 → URL 저장: {sporty_url}")
    return {"status": "fail", "path": None, "url": sporty_url}


# ──────────────────────────────────────────
# 4. 단일 선수 수집 파이프라인
# ──────────────────────────────────────────

def collect_player_videos(
    target: PlayerTarget,
    start_date: str,
    end_date: str,
    dry_run: bool = False,
) -> pd.DataFrame:

    print("\n" + "=" * 60)
    print(f"수집 시작: {target.name}")
    print(f"분석 목적: {target.analysis_focus}")
    print("=" * 60)

    # 1. Statcast
    stat_df = get_player_statcast(target, start_date, end_date)
    if stat_df.empty:
        print(f"  ✗ {target.name}: Statcast 데이터 없음 — 날짜 범위 또는 이름 확인 필요")
        return pd.DataFrame()

    # 저장 경로
    player_dir = os.path.join(VIDEO_DIR, target.name.lower())
    os.makedirs(player_dir, exist_ok=True)

    rows = []
    success_count = 0
    fail_count = 0

    for i, (_, row) in enumerate(stat_df.head(target.max_videos).iterrows()):
        play_id = resolve_play_id(row)

        print(f"\n  [{i+1}/{min(len(stat_df), target.max_videos)}] "
              f"{row.get('game_date')} | {row.get('events')} | "
              f"EV: {row.get('launch_speed', 'N/A')} mph | "
              f"구종: {row.get('pitch_name', 'N/A')}")

        if play_id is None:
            print("  ✗ play_id 없음 — 스킵")
            fail_count += 1
            continue

        print(f"  play_id: {play_id}")

        result = download_video(play_id, player_dir, dry_run=dry_run)

        # 메타데이터 수집
        meta = {
            "player":        target.name,
            "role":          target.role,
            "game_date":     row.get("game_date"),
            "events":        row.get("events"),
            "pitch_name":    row.get("pitch_name"),
            "release_speed": row.get("release_speed"),
            "release_spin_rate": row.get("release_spin_rate"),
            "launch_speed":  row.get("launch_speed"),
            "launch_angle":  row.get("launch_angle"),
            "hit_distance_sc": row.get("hit_distance_sc"),
            "estimated_ba":  row.get("estimated_ba_using_speedangle"),
            "woba_value":    row.get("woba_value"),
            "zone":          row.get("zone"),
            "plate_x":       row.get("plate_x"),
            "plate_z":       row.get("plate_z"),
            "balls":         row.get("balls"),
            "strikes":       row.get("strikes"),
            "inning":        row.get("inning"),
            "home_team":     row.get("home_team"),
            "away_team":     row.get("away_team"),
            "game_pk":       row.get("game_pk"),
            "play_id":       play_id,
            "video_path":    result["path"],
            "video_url":     result["url"],
            "download_status": result["status"],
        }
        rows.append(meta)

        if result["status"] == "ok":
            success_count += 1
        elif result["status"] == "skip":
            success_count += 1  # 이미 있는 것도 성공으로 카운트
        else:
            fail_count += 1

        # API rate limit 대응
        time.sleep(1.0 if not dry_run else 0.1)

    result_df = pd.DataFrame(rows)

    # 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"{target.name.lower()}_videos.csv")
    result_df.to_csv(csv_path, index=False)

    print(f"\n  [완료] {target.name}: 성공 {success_count} / 실패 {fail_count} / 총 {len(rows)}")
    print(f"  메타데이터 저장: {csv_path}")

    return result_df


# ──────────────────────────────────────────
# 5. 수집 결과 요약
# ──────────────────────────────────────────

def print_summary(all_results: dict[str, pd.DataFrame]):
    print("\n" + "=" * 60)
    print("수집 결과 요약")
    print("=" * 60)

    total_ok   = 0
    total_fail = 0

    for name, df in all_results.items():
        if df.empty:
            print(f"  {name:12s}: 데이터 없음")
            continue

        ok   = (df["download_status"].isin(["ok", "skip"])).sum()
        fail = (df["download_status"] == "fail").sum()
        dry  = (df["download_status"] == "dry_run").sum()

        total_ok   += ok
        total_fail += fail

        event_counts = df["events"].value_counts().to_dict()
        top_events = ", ".join(f"{k}({v})" for k, v in list(event_counts.items())[:4])

        print(f"  {name:12s}: 영상 {ok}개 확보 | 실패 {fail}개 | dry-run {dry}개")
        print(f"             이벤트 분포: {top_events}")

    print(f"\n  총계: {total_ok}개 확보 / {total_fail}개 실패")

    # 실패한 URL 목록 (수동 다운로드용)
    failed_urls = []
    for df in all_results.values():
        if df.empty:
            continue
        failed = df[df["download_status"] == "fail"][["player", "play_id", "video_url"]]
        failed_urls.append(failed)

    if failed_urls:
        failed_df = pd.concat(failed_urls)
        if not failed_df.empty:
            fail_path = os.path.join(OUTPUT_DIR, "failed_urls.txt")
            with open(fail_path, "w") as f:
                for _, r in failed_df.iterrows():
                    f.write(f"{r['player']}\t{r['play_id']}\t{r['video_url']}\n")
            print(f"\n  실패 URL 목록 저장: {fail_path}")
            print("  → yt-dlp로 수동 다운로드:")
            print(f"     yt-dlp -a {fail_path.split(chr(10))[0]} -o '%(id)s.%(ext)s'")


# ──────────────────────────────────────────
# 실행
# ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MLB 선수 영상 수집기")
    parser.add_argument(
        "--player",
        choices=["langford", "jung", "leiter", "webb", "all"],
        default="all",
        help="수집할 선수 (기본: all)",
    )
    parser.add_argument("--start", default=DEFAULT_START, help=f"시작 날짜 (기본: {DEFAULT_START})")
    parser.add_argument("--end",   default=DEFAULT_END,   help=f"종료 날짜 (기본: {DEFAULT_END})")
    parser.add_argument("--dry-run", action="store_true", help="URL만 확인, 다운로드 안 함")
    parser.add_argument("--max",   type=int, default=None, help="선수당 최대 수집 수 오버라이드")
    args = parser.parse_args()

    # 수집 대상 선택
    if args.player == "all":
        targets = TARGETS
    else:
        targets = [t for t in TARGETS if t.name.lower() == args.player]

    # max 오버라이드
    if args.max:
        for t in targets:
            t.max_videos = args.max

    print("=" * 60)
    print(f"MLB 영상 수집기")
    print(f"기간: {args.start} ~ {args.end}")
    print(f"대상: {[t.name for t in targets]}")
    if args.dry_run:
        print("모드: DRY-RUN (URL만 확인)")
    print("=" * 60)

    # yt-dlp 확인
    if not args.dry_run:
        if _yt_dlp_available():
            print("✓ yt-dlp 감지됨")
        else:
            print("⚠ yt-dlp 없음 → 직접 MP4 URL 방식만 사용")
            print("  설치 권장: pip install yt-dlp")

    all_results = {}
    for target in targets:
        df = collect_player_videos(
            target,
            start_date=args.start,
            end_date=args.end,
            dry_run=args.dry_run,
        )
        all_results[target.name] = df

    print_summary(all_results)

    # 전체 통합 CSV
    all_dfs = [df for df in all_results.values() if not df.empty]
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined_path = os.path.join(OUTPUT_DIR, "all_videos.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ 통합 메타데이터: {combined_path}")


if __name__ == "__main__":
    main()