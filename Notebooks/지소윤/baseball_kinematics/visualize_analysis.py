"""
포즈 분석 시각화 + 지표 검증
================================
1. 릴리즈 윈도우 각도 시계열 (릴리즈 포인트 = 세로선)
2. 볼넷 vs 삼진 지표 비교 박스플롯
3. 개별 프레임 스켈레톤 체크 (릴리즈 순간 포즈)

출력: pose_output/leiter/viz/ 폴더
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_analysis import calc_kinematics, THROW_HAND

WALK_DIR  = Path(__file__).parent / "pose_output/leiter"
SO_DIR    = Path(__file__).parent / "pose_output/leiter_so"
VIZ_DIR   = Path(__file__).parent / "pose_output/viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

WALK_SUMMARY = WALK_DIR / "leiter_walks_analysis.csv"
SO_SUMMARY   = SO_DIR   / "leiter_so_analysis.csv"


# ── 1. 각도 시계열 플롯 ────────────────────────────────────────

def plot_angle_series(proc_csv: str, label: str, color: str = "steelblue"):
    """
    릴리즈 윈도우 기준 각도 시계열.
    릴리즈 프레임 = release_points.csv에서 읽거나, 윈도우 중간 추정.
    """
    from release_detector import detect_release, extract_window

    play_id = Path(proc_csv).stem.replace("_trimmed_yolo_pose_proc", "")
    df = pd.read_csv(proc_csv)

    rel_frame, rel_time = detect_release(df, throw_hand=THROW_HAND, method="velocity")
    if rel_frame is None:
        return

    df_win = extract_window(df, rel_frame, pre=25, post=15)
    kin    = calc_kinematics(df_win, throw_hand=THROW_HAND)
    if kin.empty:
        return

    # frame 0 = 릴리즈
    kin["rel_frame"] = kin["frame"] - rel_frame

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"{label}  |  {play_id[:8]}...  (릴리즈 frame={rel_frame}, {rel_time:.2f}s)",
                 fontsize=13)

    pairs = [
        ("elbow_angle",    "팔꿈치 각도 (어깨-팔꿈치-손목)",  axes[0, 0]),
        ("knee_angle",     "앞발 무릎 각도",                  axes[0, 1]),
        ("shoulder_angle", "어깨 기울기 각도",                 axes[1, 0]),
        ("hip_angle",      "골반 기울기 각도",                 axes[1, 1]),
    ]

    for col, title, ax in pairs:
        s = kin[["rel_frame", col]].dropna()
        ax.plot(s["rel_frame"], s[col], color=color, lw=2)
        ax.axvline(0, color="red", lw=1.5, linestyle="--", label="릴리즈")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("릴리즈 기준 프레임")
        ax.set_ylabel("각도 (°)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = VIZ_DIR / f"{play_id[:8]}_{label}_angles.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  저장: {out.name}")


# ── 2. 릴리즈 순간 스켈레톤 체크 ──────────────────────────────

CONNECTIONS = [
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),  ("left_elbow",  "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder",  "left_hip"),    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),
    ("left_hip",       "left_knee"),   ("left_knee",  "left_ankle"),
    ("right_hip",      "right_knee"),  ("right_knee", "right_ankle"),
]

def plot_skeleton_at_release(proc_csv: str, label: str):
    """릴리즈 ±2프레임 스켈레톤 오버레이 (x/y 좌표 직접)"""
    from release_detector import detect_release, extract_window

    play_id = Path(proc_csv).stem.replace("_trimmed_yolo_pose_proc", "")
    df = pd.read_csv(proc_csv)

    rel_frame, _ = detect_release(df, throw_hand=THROW_HAND, method="velocity")
    if rel_frame is None:
        return

    df_win = extract_window(df, rel_frame, pre=2, post=2)
    df_win = df_win[df_win["detected"].fillna(False)]
    if df_win.empty:
        return

    n = len(df_win)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{label} | {play_id[:8]} — 릴리즈 ±2프레임", fontsize=12)

    joints = [
        "nose", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ]

    for ax, (_, row) in zip(axes, df_win.iterrows()):
        rel_f = int(row["frame"]) - rel_frame
        title = f"f{rel_f:+d}" + (" ← 릴리즈" if rel_f == 0 else "")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)   # y 반전 (이미지 좌표계)
        ax.set_aspect("equal")
        ax.set_facecolor("#1a1a2e")

        pts = {}
        for jname in joints:
            x = row.get(f"{jname}_x")
            y = row.get(f"{jname}_y")
            if pd.notna(x) and pd.notna(y):
                pts[jname] = (x, y)

        for a, b in CONNECTIONS:
            if a in pts and b in pts:
                ax.plot([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]],
                        color="#00e5ff", lw=2, alpha=0.8)

        for jname, (x, y) in pts.items():
            c = "#ff4444" if "wrist" in jname or "elbow" in jname else "#ffffff"
            ax.scatter(x, y, color=c, s=40, zorder=5)

    plt.tight_layout()
    out = VIZ_DIR / f"{play_id[:8]}_{label}_skeleton.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  저장: {out.name}")


# ── 3. 볼넷 vs 삼진 비교 박스플롯 ─────────────────────────────

def plot_comparison():
    if not WALK_SUMMARY.exists() or not SO_SUMMARY.exists():
        print("요약 CSV 없음 — run_analysis.py 먼저 실행")
        return

    walks = pd.read_csv(WALK_SUMMARY)
    so    = pd.read_csv(SO_SUMMARY)
    walks["label"] = "볼넷"
    so["label"]    = "삼진"
    df = pd.concat([walks, so], ignore_index=True)

    metrics = [
        ("elbow_angle_mean",  "팔꿈치 각도 평균 (°)"),
        ("knee_angle_mean",   "앞발 무릎 각도 평균 (°)"),
        ("early_open_rate",   "Early Open 비율"),
        ("arm_flyout_rate",   "Arm Flyout 비율"),
        ("sh_hp_diff_mean",   "어깨-골반 회전 차 평균 (°)"),
        ("hip_height_diff_mean", "골반 높이 차 평균"),
    ]
    metrics = [(m, t) for m, t in metrics if m in df.columns]

    n = len(metrics)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    axes = axes.flatten()
    fig.suptitle("Leiter 2025 — 볼넷 vs 삼진 키네마틱 비교", fontsize=14)

    colors = {"볼넷": "#4e9af1", "삼진": "#f1754e"}

    for ax, (metric, title) in zip(axes, metrics):
        for label, grp in df.groupby("label"):
            vals = grp[metric].dropna()
            ax.scatter([label] * len(vals), vals,
                       color=colors[label], alpha=0.6, s=60, zorder=3)
            ax.plot([label], [vals.mean()],
                    marker="D", color=colors[label], markersize=10,
                    markeredgecolor="white", zorder=5)

        # 평균 비교 텍스트
        w_mean = df[df["label"] == "볼넷"][metric].dropna().mean()
        s_mean = df[df["label"] == "삼진"][metric].dropna().mean()
        ax.set_title(f"{title}\n볼넷 {w_mean:.2f}  vs  삼진 {s_mean:.2f}", fontsize=10)
        ax.set_ylabel(title.split("(")[0].strip())
        ax.grid(True, axis="y", alpha=0.3)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    out = VIZ_DIR / "leiter_walks_vs_so_comparison.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"\n비교 플롯 저장: {out.name}")


# ── 실행 ───────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 볼넷 각도 시계열 + 스켈레톤 ===")
    walk_procs = sorted(WALK_DIR.glob("*_trimmed_yolo_pose_proc.csv"))[:5]
    for f in walk_procs:
        print(f"  {f.stem[:20]}...")
        plot_angle_series(str(f), label="walk", color="steelblue")
        plot_skeleton_at_release(str(f), label="walk")

    print("\n=== 삼진 각도 시계열 + 스켈레톤 ===")
    so_procs = sorted(SO_DIR.glob("*_trimmed_yolo_pose_proc.csv"))[:5]
    for f in so_procs:
        print(f"  {f.stem[:20]}...")
        plot_angle_series(str(f), label="so", color="tomato")
        plot_skeleton_at_release(str(f), label="so")

    print("\n=== 볼넷 vs 삼진 비교 ===")
    plot_comparison()

    print(f"\n완료 — 출력: {VIZ_DIR}")
