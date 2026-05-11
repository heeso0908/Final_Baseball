from __future__ import annotations

import bisect
from dataclasses import dataclass
import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


SEASON = 2025
TEAM_NAME = "Texas Rangers"
TEAM_ABBR = "TEX"
CACHE_VERSION = "sim-v8"

SLOT_DEFS = [
    ("C", ["C"]),
    ("1B", ["1B", "DH"]),
    ("2B", ["2B"]),
    ("3B", ["3B"]),
    ("SS", ["SS"]),
    ("LF", ["LF", "OF"]),
    ("CF", ["CF", "OF"]),
    ("RF", ["RF", "OF"]),
    ("DH", None),
]

SLOT_STARTERS = {
    "C": "Jonah Heim",
    "1B": "Jake Burger",
    "2B": "Marcus Semien",
    "3B": "Josh Jung",
    "SS": "Corey Seager",
    "LF": "Wyatt Langford",
    "RF": "Adolis Garcia",
}

MODEL_FEATURES = [
    "sv_pct",
    "SV_pg",
    "onerun_wp",
    "xi_wp",
    "home_away_diff",
    "ERA",
    "WHIP",
    "k_bb",
    "K9",
    "BB9",
    "HR9",
    "ir_pct",
    "babip_against",
    "go_ao",
    "OPS",
    "rs_per_g",
    "sb_pct",
    "era_fip_diff",
]

TEAM_CODE_TO_NAME = {
    "ARI": "Diamondbacks",
    "ATL": "Braves",
    "BAL": "Orioles",
    "BOS": "Red Sox",
    "CHC": "Cubs",
    "CHW": "White Sox",
    "CIN": "Reds",
    "CLE": "Guardians",
    "COL": "Rockies",
    "DET": "Tigers",
    "HOU": "Astros",
    "KCR": "Royals",
    "LAA": "Angels",
    "LAD": "Dodgers",
    "MIA": "Marlins",
    "MIL": "Brewers",
    "MIN": "Twins",
    "NYM": "Mets",
    "NYY": "Yankees",
    "ATH": "Athletics",
    "OAK": "Athletics",
    "PHI": "Phillies",
    "PIT": "Pirates",
    "SDP": "Padres",
    "SFG": "Giants",
    "SEA": "Mariners",
    "STL": "Cardinals",
    "TBR": "Rays",
    "TEX": "Rangers",
    "TOR": "Blue Jays",
    "WSN": "Nationals",
}


def _normalize_name(value: object) -> str:
    text = str(value).strip()
    if text.startswith("Adolis Garc"):
        return "Adolis Garcia"
    return text


def _ip_to_float(value: object) -> float:
    text = str(value)
    if "." not in text:
        return float(text)
    whole, frac = text.split(".", 1)
    try:
        return float(whole) + float(frac) / 3.0
    except ValueError:
        return float(whole)


def _coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def _safe_mean(series: pd.Series, default: float = 0.5) -> float:
    clean = series.dropna()
    return float(clean.mean()) if not clean.empty else default


def _estimate_hbp_rate(hbp_df: pd.DataFrame, batter_name: str) -> float:
    name_columns = [col for col in hbp_df.columns if col.lower() in {"name", "player", "player_name"}]
    if not name_columns:
        return 0.008
    name_col = name_columns[0]
    frame = hbp_df.copy()
    frame[name_col] = frame[name_col].map(_normalize_name)
    row = frame[frame[name_col] == batter_name]
    if row.empty:
        return 0.008
    hbp = float(row.iloc[0]["HBP"]) if "HBP" in row.columns else 0.0
    pa = float(row.iloc[0]["PA"]) if "PA" in row.columns else 0.0
    if pa > 0:
        return max(0.001, min(0.03, hbp / pa))
    return 0.008


def _extract_probs(row: pd.Series, hbp_rate: float) -> dict[str, float]:
    pa = max(float(row.get("PA", 0.0)), 1.0)
    if pd.notna(row.get("sc_pa", np.nan)):
        singles = float(row.get("sc_single_rate", 0.0)) * pa
        doubles = float(row.get("sc_double_rate", 0.0)) * pa
        triples = float(row.get("sc_triple_rate", 0.0)) * pa
        hr = float(row.get("sc_hr_rate", 0.0)) * pa
        walks = float(row.get("sc_bb_rate", 0.0)) * pa
        strikeouts = float(row.get("sc_k_rate", 0.0)) * pa
        hbp_rate = float(row.get("sc_hbp_rate", hbp_rate))
    else:
        hits = float(row.get("H", 0.0))
        doubles = float(row.get("2B", 0.0))
        triples = float(row.get("3B", 0.0))
        hr = float(row.get("HR", 0.0))
        singles = max(0.0, hits - doubles - triples - hr)
        walks = float(row.get("BB", 0.0))
        strikeouts = float(row.get("SO", 0.0))

    probs = {
        "single": singles / pa,
        "double": doubles / pa,
        "triple": triples / pa,
        "hr": hr / pa,
        "bb": walks / pa,
        "hbp": hbp_rate,
        "k": strikeouts / pa,
    }
    on_base = sum(probs.values())
    remaining_outs = max(0.01, 1.0 - on_base)
    gb_pct = float(row.get("GB %", 44.0))
    ground_share = min(0.75, max(0.30, gb_pct / 100.0))
    probs["groundout"] = remaining_outs * ground_share
    probs["flyout"] = max(0.005, remaining_outs - probs["groundout"])
    total = sum(probs.values())
    return {key: value / total for key, value in probs.items()}


def _apply_batter_boost(
    probs: dict[str, float],
    hr_mult: float = 1.0,
    double_mult: float = 1.0,
    single_mult: float = 1.0,
    bb_mult: float = 1.0,
    k_mult: float = 1.0,
) -> dict[str, float]:
    boosted = probs.copy()
    boosted["hr"] *= hr_mult
    boosted["double"] *= double_mult
    boosted["single"] *= single_mult
    boosted["bb"] *= bb_mult
    boosted["k"] *= k_mult
    total = sum(boosted.values())
    return {key: value / total for key, value in boosted.items()}


def _adjust_probs_for_pitcher(
    probs: dict[str, float],
    k_factor: float = 1.0,
    bb_factor: float = 1.0,
    hr_factor: float = 1.0,
    hit_factor: float = 1.0,
) -> dict[str, float]:
    adjusted = probs.copy()
    for key in ("k",):
        adjusted[key] *= k_factor
    for key in ("bb", "hbp"):
        adjusted[key] *= bb_factor
    adjusted["hr"] *= hr_factor
    for key in ("single", "double", "triple"):
        adjusted[key] *= hit_factor
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def _scale_probs_for_park(probs: dict[str, float], park: dict[str, float]) -> dict[str, float]:
    adjusted = probs.copy()
    for key in ("single", "double", "triple", "hr", "bb"):
        if key in adjusted and key in park:
            adjusted[key] *= float(park[key])
    total = sum(adjusted.values())
    return {key: value / total for key, value in adjusted.items()}


def _build_profile(row: pd.Series, sprint_map: dict[str, float]) -> dict[str, float | str]:
    name = _normalize_name(row["Name"])
    sprint = float(sprint_map.get(name, 27.0))
    gb_pct = float(row.get("GB %", 44.0))
    fb_pct = float(row.get("FB %", 26.0))
    return {
        "name": name,
        "speed": sprint,
        "gdp_rate": max(0.02, min(0.20, gb_pct / 300.0)),
        "sf_rate": max(0.01, min(0.10, fb_pct / 350.0)),
    }


def _advance_runners(
    occupied: list[int],
    outs: int,
    event: str,
    profile: dict[str, float | str],
    advancement: AdvancementTables | None = None,
) -> tuple[list[int], int, int]:
    runs = 0
    speed = float(profile.get("speed", 27.0))
    aggressive = speed >= 28.5

    if event in {"k", "groundout", "flyout"}:
        if event == "groundout" and 1 in occupied and outs <= 1:
            if np.random.random() < float(profile.get("gdp_rate", 0.08)):
                outs += 2
                occupied = [base for base in occupied if base != 1]
                return occupied, min(3, outs), runs
        if event == "flyout" and 3 in occupied and outs <= 1:
            if np.random.random() < float(profile.get("sf_rate", 0.04)):
                runs += 1
                occupied = [base for base in occupied if base != 3]
        outs += 1
        return occupied, min(3, outs), runs

    if event in {"bb", "hbp"}:
        bases = set(occupied)
        if {1, 2, 3}.issubset(bases):
            runs += 1
        new_bases = set()
        if 3 in bases:
            new_bases.add(3)
        if 2 in bases:
            new_bases.add(3 if 1 in bases else 2)
        if 1 in bases:
            new_bases.add(2)
        new_bases.add(1)
        return sorted(new_bases), outs, runs

    if event == "single":
        if advancement is not None:
            state = tuple(sorted(occupied))
            options = advancement.single.get(state)
            if options:
                idx = int(np.random.choice(len(options)))
                next_bases, scored_runs = options[idx]
                return list(next_bases), outs, int(scored_runs)
        new_bases = [1]
        for base in sorted(occupied, reverse=True):
            if base == 3:
                runs += 1
            elif base == 2:
                if aggressive or np.random.random() < 0.65:
                    runs += 1
                else:
                    new_bases.append(3)
            elif base == 1:
                new_bases.append(3 if aggressive or np.random.random() < 0.45 else 2)
        return sorted(set(new_bases)), outs, runs

    if event == "double":
        if advancement is not None:
            state = tuple(sorted(occupied))
            options = advancement.double.get(state)
            if options:
                idx = int(np.random.choice(len(options)))
                next_bases, scored_runs = options[idx]
                return list(next_bases), outs, int(scored_runs)
        new_bases = [2]
        for base in sorted(occupied, reverse=True):
            if base in {2, 3}:
                runs += 1
            elif base == 1:
                if aggressive or np.random.random() < 0.60:
                    runs += 1
                else:
                    new_bases.append(3)
        return sorted(set(new_bases)), outs, runs

    if event == "triple":
        runs += len(occupied)
        return [3], outs, runs

    if event == "hr":
        runs += len(occupied) + 1
        return [], outs, runs

    return occupied, outs, runs


@dataclass
class ModelBundle:
    scaler: StandardScaler
    ridge: Ridge
    lasso: Lasso
    rf: RandomForestRegressor
    boost_model: object
    ensemble_cv_mae: float
    tex25: dict[str, float]
    feature_means: pd.Series


@dataclass
class BullpenState:
    inning_era_map: dict[int, float]
    team_era_actual: float
    starter_era_actual: float
    starter_ip_mean: float
    starter_ip_std: float
    closer_era: float
    setup_era: float
    bullpen_era: float
    reliever_stats: pd.DataFrame
    closer_names: set[str]
    closer_pool_p1: list[str]   # Mar-Apr  (games 0-30):  Jackson
    closer_pool_p2: list[str]   # May-Jul  (games 31-109): Garcia 주, Armstrong 보조
    closer_pool_p3: list[str]   # Aug-Sep  (games 110+):  Armstrong 주, Garcia 보조


def _base_state_from_row(row: pd.Series, prefix: str = "on") -> tuple[int, ...]:
    occupied = []
    for base, column in [(1, f"{prefix}_1b"), (2, f"{prefix}_2b"), (3, f"{prefix}_3b")]:
        value = row.get(column, "")
        if pd.notna(value) and str(value).strip() != "":
            occupied.append(base)
    return tuple(sorted(occupied))


@dataclass
class AdvancementTables:
    single: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]]
    double: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]]


def _bundle_cache_paths(raw_dir: Path) -> tuple[Path, Path]:
    cache_dir = raw_dir.parent / ".cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir, cache_dir / "model_bundle.pkl"


def _bundle_signature(raw_dir: Path) -> str:
    watched_files = [
        "mlb_team_seasons.csv",
        "texas_2025_game_log.csv",
        "tex_2025_save_situation_splits.csv",
        "rangers_pitcher_gamelogs.csv",
    ]
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    for name in watched_files:
        path = raw_dir / name
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _load_cached_bundle(raw_dir: Path) -> ModelBundle | None:
    _, cache_file = _bundle_cache_paths(raw_dir)
    if not cache_file.exists():
        return None
    try:
        with cache_file.open("rb") as handle:
            payload = pickle.load(handle)
        if payload.get("signature") != _bundle_signature(raw_dir):
            return None
        bundle = payload.get("bundle")
        if isinstance(bundle, ModelBundle):
            return bundle
    except Exception:
        return None
    return None


def _save_cached_bundle(raw_dir: Path, bundle: ModelBundle) -> None:
    _, cache_file = _bundle_cache_paths(raw_dir)
    payload = {
        "signature": _bundle_signature(raw_dir),
        "bundle": bundle,
    }
    with cache_file.open("wb") as handle:
        pickle.dump(payload, handle)


def _advancement_cache_file(raw_dir: Path) -> Path:
    cache_dir, _ = _bundle_cache_paths(raw_dir)
    watched_files = [
        "statcast_tex_singles_2025.csv",
        "statcast_tex_doubles_2025.csv",
    ]
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    for name in watched_files:
        path = raw_dir / name
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return cache_dir / f"advancement_{digest.hexdigest()[:16]}.pkl"


def _simulation_cache_file(raw_dir: Path, scenario_name: str, n_sims: int) -> Path:
    cache_dir, _ = _bundle_cache_paths(raw_dir)
    watched_files = [
        "batting_stats_2025_all.csv",
        "rangers_roster_2025.csv",
        "rangers_hitting_batted.csv",
        "rangers_running.csv",
        "hit_by_pitch.csv",
        "statcast_tex_all_events_2025.csv",
        "texas_pitchers_2025.csv",
        "mlb_teams_2025_pitching.csv",
        "texas_2025_game_log.csv",
        "fangraphs-guts-data.csv",
        "tex_2025_pitching_inning_splits.csv",
        "tex_2025_save_situation_splits.csv",
        "rangers_pitcher_gamelogs.csv",
    ]
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    digest.update(_bundle_signature(raw_dir).encode("utf-8"))
    digest.update(scenario_name.encode("utf-8"))
    digest.update(str(n_sims).encode("utf-8"))
    for name in watched_files:
        path = raw_dir / name
        stat = path.stat()
        digest.update(name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return cache_dir / f"simulation_{digest.hexdigest()[:16]}.pkl"


class LineupPool:
    def __init__(
        self,
        pool_df: pd.DataFrame,
        sprint_map: dict[str, float],
        hbp_df: pd.DataFrame,
        name_boosts: dict[str, dict[str, float]] | None = None,
        team_boosts: dict[str, float] | None = None,
    ) -> None:
        self.probs: dict[str, tuple[list, list]] = {}
        self.profiles: dict[str, dict[str, float | str]] = {}
        self.obp: dict[str, float] = {}
        self.start_weight: dict[str, float] = {}
        self.eligibility: dict[str, list[str]] = {}

        for _, row in pool_df.iterrows():
            name = _normalize_name(row["Name"])
            probs = _extract_probs(row, _estimate_hbp_rate(hbp_df, name))
            if team_boosts:
                probs = _apply_batter_boost(probs, **team_boosts)
            if name_boosts and name in name_boosts:
                probs = _apply_batter_boost(probs, **name_boosts[name])
            # (events, cum_weights) 튜플로 미리 변환 — 타석 루프에서 list() 생성 제거
            _evts = list(probs.keys())
            _cum: list[float] = []
            _total = 0.0
            for w in probs.values():
                _total += float(w)
                _cum.append(_total)
            self.probs[name] = (_evts, _cum)
            self.profiles[name] = _build_profile(row, sprint_map)
            self.obp[name] = float(row.get("OBP", 0.320) or 0.320)
            self.start_weight[name] = float(row["start_w"])
            self.eligibility[name] = list(row["elig_pos"])

        self.names = list(self.probs.keys())
        self.slot_pools: list[list[str] | None] = []
        for _, positions in SLOT_DEFS:
            if positions is None:
                self.slot_pools.append(None)
            else:
                candidates = [
                    name for name in self.names
                    if any(pos in positions for pos in self.eligibility[name])
                ]
                self.slot_pools.append(candidates)

    def sample(self) -> tuple[list[dict[str, float]], list[dict[str, float | str]], list[str]]:
        selected: list[str] = []
        used: set[str] = set()

        for index, (slot_name, _) in enumerate(SLOT_DEFS):
            starter = SLOT_STARTERS.get(slot_name)
            if starter and starter in self.start_weight and starter not in used:
                if np.random.random() < self.start_weight[starter]:
                    selected.append(starter)
                    used.add(starter)
                    continue

            pool = self.slot_pools[index] if self.slot_pools[index] is not None else self.names
            available = [name for name in pool if name not in used]
            if starter:
                available = [name for name in available if name != starter]
            if not available:
                available = [name for name in self.names if name not in used]
            if not available:
                continue

            weights = np.array([self.start_weight[name] for name in available], dtype=float)
            weights = weights / weights.sum()
            chosen = available[int(np.random.choice(len(available), p=weights))]
            selected.append(chosen)
            used.add(chosen)

        selected = sorted(selected, key=lambda name: self.obp[name], reverse=True)
        return (
            [self.probs[name] for name in selected],
            [self.profiles[name] for name in selected],
            selected,
        )


def _simulate_inning(
    lineup_probs: list[dict[str, float]],
    profiles: list[dict[str, float | str]],
    start_index: int,
    advancement: AdvancementTables | None = None,
) -> tuple[int, int]:
    occupied: list[int] = []
    outs = 0
    runs = 0
    hitter_index = start_index
    lineup_size = len(lineup_probs)

    while outs < 3:
        slot = hitter_index % lineup_size
        _evts, _cum = lineup_probs[slot]
        event = _evts[bisect.bisect(_cum, np.random.random())]
        occupied, outs, scored = _advance_runners(occupied, outs, event, profiles[slot], advancement=advancement)
        runs += scored
        hitter_index += 1

    return runs, hitter_index % lineup_size


def _simulate_texas_runs(
    lineup_probs: list[dict[str, float]],
    profiles: list[dict[str, float | str]],
    advancement: AdvancementTables | None = None,
) -> int:
    total_runs = 0
    lineup_index = 0
    for _ in range(9):
        inning_runs, lineup_index = _simulate_inning(lineup_probs, profiles, lineup_index, advancement=advancement)
        total_runs += inning_runs
    return total_runs


def _choose_bullpen_pitcher(
    bullpen: BullpenState,
    era_scale: float,
    inning: int,
    game_index: int,
    fraction: float,
    park_factor: float,
    opp_ratio: float,
    tracker: dict[str, dict[str, float]],
    recent_apps: dict[str, list[int]],
    lead: int | None,
) -> float:
    relievers = bullpen.reliever_stats.copy()
    relievers = relievers[relievers["IP"] >= 5.0].copy()

    if inning >= 9 and lead is not None and 1 <= lead <= 3:
        if game_index <= 30:
            closer_names = bullpen.closer_pool_p1    # Mar-Apr: Jackson
        elif game_index <= 109:
            closer_names = bullpen.closer_pool_p2    # May-Jul: Garcia 주
        else:
            closer_names = bullpen.closer_pool_p3    # Aug-Sep: Armstrong 주
        pool = relievers[relievers["Name"].isin(closer_names)].copy()
        if pool.empty:
            pool = relievers.copy()
    elif inning >= 7:
        pool = relievers.copy()
    else:
        pool = relievers[~relievers["Name"].isin(bullpen.closer_names)].copy()
        if pool.empty:
            pool = relievers.copy()

    available_rows = []
    weights = []
    for _, row in pool.iterrows():
        name = str(row["Name"])
        if recent_apps.get(name) == [game_index - 2, game_index - 1]:
            continue
        usage_ip = tracker.get(name, {}).get("IP", 0.0)
        decay = max(0.10, 1.0 - (usage_ip / 80.0) ** 2)
        weight = max(0.01, float(row["usage_weight"]) * decay)
        available_rows.append(row)
        weights.append(weight)

    if not available_rows:
        if inning >= 9 and lead is not None and 1 <= lead <= 3:
            return bullpen.closer_era * era_scale
        return (bullpen.setup_era if inning >= 7 else bullpen.bullpen_era) * era_scale

    probs = np.array(weights, dtype=float)
    probs = probs / probs.sum()
    chosen = available_rows[int(np.random.choice(len(available_rows), p=probs))]
    name = str(chosen["Name"])
    tracker.setdefault(name, {"IP": 0.0})
    tracker[name]["IP"] += fraction
    history = sorted(set(recent_apps.get(name, []) + [game_index]))[-2:]
    recent_apps[name] = history
    return float(chosen["ERA"]) * era_scale


def _simulate_opponent_runs(
    bullpen: BullpenState,
    scenario_era: float,
    starter_era: float,
    starter_ip: float,
    park_factor: float,
    opp_ratio: float,
    game_index: int,
    tracker: dict[str, dict[str, float]],
    recent_apps: dict[str, list[int]],
    texas_runs_hint: int | None = None,
) -> int:
    era_scale = scenario_era / max(0.1, bullpen.team_era_actual)
    sampled_ip = float(np.clip(np.random.normal(starter_ip, max(0.25, bullpen.starter_ip_std)), 0.0, 9.0))
    full_starter_innings = int(sampled_ip)
    partial_inning = sampled_ip - full_starter_innings
    total_runs = 0

    for inning in range(1, 10):
        if inning <= full_starter_innings:
            base_era = bullpen.inning_era_map.get(inning, bullpen.starter_era_actual) * era_scale
            relative = starter_era / max(0.1, bullpen.starter_era_actual)
            mu = max(1e-9, base_era * relative / 9.0 * park_factor * opp_ratio)
            total_runs += int(np.random.poisson(mu))
            continue

        if inning == full_starter_innings + 1 and partial_inning > 0:
            base_era = bullpen.inning_era_map.get(inning, bullpen.starter_era_actual) * era_scale
            relative = starter_era / max(0.1, bullpen.starter_era_actual)
            sp_mu = max(1e-9, base_era * relative * partial_inning / 9.0 * park_factor * opp_ratio)
            total_runs += int(np.random.poisson(sp_mu))
            bullpen_fraction = 1.0 - partial_inning
        else:
            bullpen_fraction = 1.0

        lead = None if texas_runs_hint is None else texas_runs_hint - total_runs
        reliever_era = _choose_bullpen_pitcher(
            bullpen=bullpen,
            era_scale=era_scale,
            inning=inning,
            game_index=game_index,
            fraction=bullpen_fraction,
            park_factor=park_factor,
            opp_ratio=opp_ratio,
            tracker=tracker,
            recent_apps=recent_apps,
            lead=lead,
        )
        bp_mu = max(1e-9, reliever_era * bullpen_fraction / 9.0 * park_factor * opp_ratio)
        total_runs += int(np.random.poisson(bp_mu))

    return total_runs


def _clean_schedule(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["Opp"] = frame["Opp"].astype(str).str.strip()
    frame["Home_Away"] = frame["Home_Away"].astype(str).str.strip()
    frame["R"] = pd.to_numeric(frame["R"], errors="coerce")
    frame["RA"] = pd.to_numeric(frame["RA"], errors="coerce")
    cleaned_dates = frame["Date"].astype(str).str.replace(r"\s+\(\d+\)$", "", regex=True)
    frame["date"] = pd.to_datetime(cleaned_dates + f", {SEASON}", format="%A, %b %d, %Y", errors="coerce")
    frame["month"] = frame["date"].dt.strftime("%b")
    frame["win"] = frame["W/L"].astype(str).str.startswith("W")
    frame["is_home"] = frame["Home_Away"].eq("Home")
    frame["series_break"] = (
        (frame["Opp"] != frame["Opp"].shift(1))
        | (frame["is_home"] != frame["is_home"].shift(1))
    ).fillna(True)
    frame["series_id"] = frame["series_break"].cumsum().astype(int)
    frame["run_diff"] = frame["R"] - frame["RA"]
    frame["onerun"] = frame["run_diff"].abs() == 1
    innings = pd.to_numeric(frame["Inn"], errors="coerce").fillna(9.0)
    frame["extra"] = innings > 9.0
    return frame


def _build_tex25(raw_dir: Path) -> dict[str, float]:
    team_hist = pd.read_csv(raw_dir / "mlb_team_seasons.csv")
    team_hist = _coerce_numeric(team_hist, ["year", "ir_pct", "babip_against", "go_ao", "OPS", "sb_pct", "era_fip_diff"])
    team_row = team_hist[(team_hist["year"] == SEASON) & (team_hist["team"] == TEAM_NAME)].iloc[0]

    schedule = _clean_schedule(pd.read_csv(raw_dir / "texas_2025_game_log.csv"))
    save_df = pd.read_csv(raw_dir / "tex_2025_save_situation_splits.csv")
    save_df = _coerce_numeric(save_df, ["SV", "BS"])
    save_total = save_df[save_df["Name"] == "Team Total"].iloc[0]

    pitcher_logs = pd.read_csv(raw_dir / "rangers_pitcher_gamelogs.csv")
    pitcher_logs = _coerce_numeric(pitcher_logs, ["ER", "H", "BB", "SO", "HR"])
    ip_total = pitcher_logs["IP"].map(_ip_to_float).sum()
    team_era_actual = float(pitcher_logs["ER"].sum() * 9.0 / max(1.0, ip_total))

    rs = float(schedule["R"].sum())
    ra = float(schedule["RA"].sum())
    games = float(len(schedule))
    pyth_wp = rs**1.83 / max(1.0, rs**1.83 + ra**1.83)

    saves = float(save_total["SV"])
    blown_saves = float(save_total["BS"])
    save_opps = max(1.0, saves + blown_saves)

    return {
        "W": float(schedule["win"].sum()),
        "L": float((~schedule["win"]).sum()),
        "G": games,
        "RS": rs,
        "RA": ra,
        "home_wp": _safe_mean(schedule[schedule["is_home"]]["win"]),
        "away_wp": _safe_mean(schedule[~schedule["is_home"]]["win"]),
        "onerun_wp": _safe_mean(schedule[schedule["onerun"]]["win"]),
        "onerun_n": float(schedule["onerun"].sum()),
        "xi_wp": _safe_mean(schedule[schedule["extra"]]["win"], 0.5),
        "SV": saves,
        "BS": blown_saves,
        "SVO": save_opps,
        "sv_pct": saves / save_opps,
        "ERA": float(pitcher_logs["ER"].sum() * 9.0 / max(1.0, ip_total)),
        "WHIP": float((pitcher_logs["H"].sum() + pitcher_logs["BB"].sum()) / max(1.0, ip_total)),
        "K9": float(pitcher_logs["SO"].sum() * 9.0 / max(1.0, ip_total)),
        "BB9": float(pitcher_logs["BB"].sum() * 9.0 / max(1.0, ip_total)),
        "HR9": float(pitcher_logs["HR"].sum() * 9.0 / max(1.0, ip_total)),
        "ir_pct": float(team_row["ir_pct"]),
        "babip_against": float(team_row["babip_against"]),
        "go_ao": float(team_row["go_ao"]),
        "OPS": float(team_row["OPS"]),
        "sb_pct": float(team_row["sb_pct"]),
        "era_fip_diff": float(team_row["era_fip_diff"]),
        "home_away_diff": _safe_mean(schedule[schedule["is_home"]]["win"]) - _safe_mean(schedule[~schedule["is_home"]]["win"]),
        "rs_per_g": rs / max(1.0, games),
        "SV_pg": saves / max(1.0, games),
        "pyth_W": pyth_wp * games,
        "residual": float(schedule["win"].sum()) - (pyth_wp * games),
    }


def _train_model_bundle(raw_dir: Path) -> ModelBundle:
    cached = _load_cached_bundle(raw_dir)
    if cached is not None:
        return cached

    hist = pd.read_csv(raw_dir / "mlb_team_seasons.csv")
    hist = _coerce_numeric(
        hist,
        [
            "year", "W", "L", "G", "RS", "RA", "home_wp", "away_wp", "onerun_wp", "onerun_n", "xi_wp",
            "ERA", "WHIP", "K9", "BB9", "SV", "SVO", "BS", "sv_pct", "HR9", "babip_against",
            "go_ao", "ir_pct", "OPS", "sb_pct", "era_fip_diff",
        ],
    )
    hist = hist[(hist["year"] < SEASON) & (hist["year"] != 2020)].copy()
    hist["pyth_wp"] = hist["RS"] ** 1.83 / (hist["RS"] ** 1.83 + hist["RA"] ** 1.83)
    hist["pyth_W"] = hist["pyth_wp"] * hist["G"]
    hist["residual"] = hist["W"] - hist["pyth_W"]
    hist["home_away_diff"] = hist["home_wp"] - hist["away_wp"]
    hist["k_bb"] = hist["K9"] / hist["BB9"].clip(lower=0.1)
    hist["rs_per_g"] = hist["RS"] / hist["G"]
    hist["SV_pg"] = hist["SV"] / hist["G"]

    tex25 = _build_tex25(raw_dir)
    df_model = hist.dropna(subset=MODEL_FEATURES + ["residual"]).copy()
    X_raw = df_model[MODEL_FEATURES].values
    y = df_model["residual"].values
    groups = df_model["year"].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    group_kfold = GroupKFold(n_splits=df_model["year"].nunique())
    splits = list(group_kfold.split(X_scaled, y, groups))

    alphas = np.logspace(-2, 3, 40)
    ridge_alpha = alphas[np.argmax([
        cross_val_score(Ridge(alpha=a), X_scaled, y, cv=splits, scoring="r2").mean()
        for a in alphas
    ])]
    lasso_alpha = alphas[np.argmax([
        cross_val_score(Lasso(alpha=a, max_iter=5000), X_scaled, y, cv=splits, scoring="r2").mean()
        for a in alphas
    ])]

    ridge = Ridge(alpha=float(ridge_alpha)).fit(X_scaled, y)
    lasso = Lasso(alpha=float(lasso_alpha), max_iter=5000).fit(X_scaled, y)
    rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42).fit(X_raw, y)
    if XGBRegressor is not None:
        boost_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            random_state=42,
            verbosity=0,
        ).fit(X_raw, y)
        boost_pipe = XGBRegressor(
            n_estimators=100,
            learning_rate=0.03,
            max_depth=2,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=3,
            random_state=42,
            verbosity=0,
        )
    else:
        boost_model = GradientBoostingRegressor(random_state=42).fit(X_raw, y)
        boost_pipe = GradientBoostingRegressor(random_state=42)

    pipes = [
        make_pipeline(StandardScaler(), Ridge(alpha=ridge.alpha)),
        make_pipeline(StandardScaler(), Lasso(alpha=lasso.alpha, max_iter=5000)),
        RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=42),
        boost_pipe,
    ]
    maes = [
        -cross_val_score(pipe, X_raw, y, cv=splits, scoring="neg_mean_absolute_error").mean()
        for pipe in pipes
    ]

    bundle = ModelBundle(
        scaler=scaler,
        ridge=ridge,
        lasso=lasso,
        rf=rf,
        boost_model=boost_model,
        ensemble_cv_mae=float(np.mean(maes)),
        tex25=tex25,
        feature_means=df_model[MODEL_FEATURES].mean(),
    )
    _save_cached_bundle(raw_dir, bundle)
    return bundle


def _predict_residual(bundle: ModelBundle, stats: dict[str, float]) -> float:
    feature_row = {**bundle.tex25, **stats}
    feature_row["k_bb"] = feature_row["K9"] / max(0.1, feature_row["BB9"])
    feature_row["home_away_diff"] = feature_row.get("home_away_diff", bundle.tex25["home_away_diff"])
    feature_row["SV_pg"] = feature_row.get("SV_pg", bundle.tex25["SV_pg"])

    if all(key in stats for key in ("K9", "BB9", "HR9")):
        base_fip = bundle.tex25["ERA"] - bundle.tex25["era_fip_diff"]
        constant = base_fip - (13 * bundle.tex25["HR9"] - 2 * bundle.tex25["K9"] + 3 * bundle.tex25["BB9"]) / 9.0
        estimated_fip = constant + (13 * feature_row["HR9"] - 2 * feature_row["K9"] + 3 * feature_row["BB9"]) / 9.0
        feature_row["era_fip_diff"] = feature_row["ERA"] - estimated_fip
    else:
        feature_row["era_fip_diff"] = bundle.tex25["era_fip_diff"]

    vector = np.array([[feature_row.get(feature, float(bundle.feature_means[feature])) for feature in MODEL_FEATURES]])
    vector_scaled = bundle.scaler.transform(vector)
    predictions = [
        bundle.ridge.predict(vector_scaled)[0],
        bundle.lasso.predict(vector_scaled)[0],
        bundle.rf.predict(vector)[0],
        bundle.boost_model.predict(vector)[0],
    ]
    return float(np.mean(predictions))


def _build_scenarios(tex25: dict[str, float]) -> dict[str, dict[str, object]]:
    return {
        "Baseline 2025": {
            "stats": {
                "ERA": tex25["ERA"],
                "sv_pct": tex25["sv_pct"],
                "onerun_wp": tex25["onerun_wp"],
                "xi_wp": tex25["xi_wp"],
                "K9": tex25["K9"],
                "BB9": tex25["BB9"],
                "HR9": tex25["HR9"],
                "WHIP": tex25["WHIP"],
            },
            "boosts": None,
        },
        "Bullpen Upgrade": {
            "stats": {
                "ERA": tex25["ERA"],
                "sv_pct": 0.700,
                "onerun_wp": min(tex25["onerun_wp"] + 0.030, 0.600),
                "xi_wp": tex25["xi_wp"],
                "K9": tex25["K9"],
                "BB9": tex25["BB9"],
                "HR9": tex25["HR9"],
                "WHIP": tex25["WHIP"],
            },
            "boosts": None,
        },
        "Hitter Boost": {
            "stats": {
                "ERA": tex25["ERA"],
                "sv_pct": tex25["sv_pct"],
                "onerun_wp": tex25["onerun_wp"],
                "xi_wp": tex25["xi_wp"],
                "K9": tex25["K9"],
                "BB9": tex25["BB9"],
                "HR9": tex25["HR9"],
                "WHIP": tex25["WHIP"],
            },
            "boosts": None,
        },
    }


def _build_batting_pool(raw_dir: Path) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    batting = pd.read_csv(raw_dir / "batting_stats_2025_all.csv")
    batting = _coerce_numeric(batting, ["G", "PA", "AB", "H", "2B", "3B", "HR", "BB", "SO", "OBP", "SF", "SH"])
    batting = batting[batting["Tm"] == "Texas"].copy()
    batting["Name"] = batting["Name"].map(_normalize_name)

    roster = pd.read_csv(raw_dir / "rangers_roster_2025.csv")
    roster = roster[roster["type"] == "batter"].copy()
    roster["name"] = roster["name"].map(_normalize_name)
    batting = batting.merge(roster[["name", "pos"]], left_on="Name", right_on="name", how="left")
    batting["Pos"] = batting["pos"].fillna("UT")

    batted = pd.read_csv(raw_dir / "rangers_hitting_batted.csv")
    batted["Player"] = batted["Player"].map(_normalize_name)
    batted = _coerce_numeric(batted, ["GB %", "FB %"])
    batting = batting.merge(batted[["Player", "GB %", "FB %"]], left_on="Name", right_on="Player", how="left")

    batting["start_w"] = (batting["G"] / 162.0).clip(0.01, 1.0)
    batting["elig_pos"] = batting["Pos"].apply(lambda value: [str(value)])
    batting.loc[batting["Name"] == "Corey Seager", "start_w"] = 0.65
    batting.loc[batting["Name"] == "Josh Smith", "start_w"] = 0.50
    batting.loc[batting["Name"] == "Jonah Heim", "start_w"] = round(4 / 7, 4)
    batting.loc[batting["Name"] == "Kyle Higashioka", "start_w"] = round(3 / 7, 4)

    running = pd.read_csv(raw_dir / "rangers_running.csv")
    running = _coerce_numeric(running, ["Sprint Speed (ft/s)"])
    sprint_map = {
        _normalize_name(row["Player"]): float(row["Sprint Speed (ft/s)"])
        for _, row in running.iterrows()
    }
    statcast_events = pd.read_csv(raw_dir / "statcast_tex_all_events_2025.csv")
    statcast_events["batter"] = pd.to_numeric(statcast_events["batter"], errors="coerce")
    statcast_events["events"] = statcast_events["events"].astype(str).str.strip()
    event_map = {
        "single": "single",
        "double": "double",
        "triple": "triple",
        "home_run": "hr",
        "walk": "bb",
        "intent_walk": "bb",
        "hit_by_pitch": "hbp",
        "strikeout": "k",
        "strikeout_double_play": "k",
        "field_error": "single",
        "force_out": "groundout",
        "fielders_choice": "groundout",
        "fielders_choice_out": "groundout",
        "grounded_into_double_play": "groundout",
        "double_play": "groundout",
        "sac_bunt": "groundout",
        "sac_fly": "flyout",
        "field_out": "flyout",
    }
    statcast_events["event_bucket"] = statcast_events["events"].map(event_map)
    statcast_events = statcast_events[statcast_events["event_bucket"].notna()].copy()
    roster_ids = roster[["name", "player_id"]].copy()
    roster_ids["name"] = roster_ids["name"].map(_normalize_name)
    roster_ids["player_id"] = pd.to_numeric(roster_ids["player_id"], errors="coerce")
    event_rates = (
        statcast_events.groupby(["batter", "event_bucket"]).size().unstack(fill_value=0).reset_index()
    )
    event_rates["sc_pa"] = event_rates[[col for col in event_rates.columns if col != "batter"]].sum(axis=1)
    for source, target in [
        ("single", "sc_single_rate"),
        ("double", "sc_double_rate"),
        ("triple", "sc_triple_rate"),
        ("hr", "sc_hr_rate"),
        ("bb", "sc_bb_rate"),
        ("hbp", "sc_hbp_rate"),
        ("k", "sc_k_rate"),
    ]:
        event_rates[target] = event_rates.get(source, 0) / event_rates["sc_pa"].clip(lower=1.0)
    batting["mlbID"] = pd.to_numeric(batting["mlbID"], errors="coerce")
    batting = batting.merge(event_rates, left_on="mlbID", right_on="batter", how="left")
    hbp_df = pd.read_csv(raw_dir / "hit_by_pitch.csv")
    return batting[batting["PA"] >= 20].copy(), sprint_map, hbp_df


def _build_advancement_tables(raw_dir: Path) -> AdvancementTables:
    cache_file = _advancement_cache_file(raw_dir)
    if cache_file.exists():
        try:
            with cache_file.open("rb") as handle:
                cached = pickle.load(handle)
            if isinstance(cached, AdvancementTables):
                return cached
        except Exception:
            pass

    def load_event_table(filename: str, batter_base: int) -> dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]]:
        df = pd.read_csv(raw_dir / filename)
        table: dict[tuple[int, ...], list[tuple[tuple[int, ...], int]]] = {}
        for _, row in df.iterrows():
            pre_state = _base_state_from_row(row, prefix="on")
            post_state = _base_state_from_row(row, prefix="post_on")
            total_before = len(pre_state) + 1
            total_after = len(post_state)
            runs_scored = max(0, total_before - total_after)
            if not pre_state and not post_state:
                post_state = (batter_base,)
            table.setdefault(pre_state, []).append((post_state, runs_scored))
        return table

    tables = AdvancementTables(
        single=load_event_table("statcast_tex_singles_2025.csv", batter_base=1),
        double=load_event_table("statcast_tex_doubles_2025.csv", batter_base=2),
    )
    with cache_file.open("wb") as handle:
        pickle.dump(tables, handle)
    return tables


def _build_starters(raw_dir: Path) -> pd.DataFrame:
    starters = pd.read_csv(raw_dir / "texas_pitchers_2025.csv")
    starters = _coerce_numeric(starters, ["GS", "ERA", "IP"])
    starters = starters[(starters["Team"] == TEAM_ABBR) & (starters["GS"] > 0)].copy()
    starters["Name"] = starters["Name"].map(_normalize_name)
    starters["avg_ip"] = starters["IP"] / starters["GS"].clip(lower=1.0)
    starters["start_w"] = (starters["GS"] / 162.0).clip(0.01, 1.0)
    return starters.sort_values("GS", ascending=False).reset_index(drop=True)


def _lineup_probs_as_dict(probs_tuple: tuple) -> dict[str, float]:
    """(events, cum_weights) 튜플 → {event: prob} dict 복원."""
    evts, cum = probs_tuple
    weights = [cum[0]] + [cum[i] - cum[i - 1] for i in range(1, len(cum))]
    return dict(zip(evts, weights))


def _build_player_projection_table(
    batting_pool: pd.DataFrame,
    sprint_map: dict[str, float],
    hbp_df: pd.DataFrame,
    scenario_boosts: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    baseline_pool = LineupPool(batting_pool, sprint_map, hbp_df, None)
    scenario_pool = LineupPool(batting_pool, sprint_map, hbp_df, scenario_boosts)
    player_rows: list[dict[str, float | str]] = []

    for _, row in batting_pool.iterrows():
        name = _normalize_name(row["Name"])
        base_probs = _lineup_probs_as_dict(baseline_pool.probs[name])
        sim_probs = _lineup_probs_as_dict(scenario_pool.probs[name])
        pos = str(row.get("Pos", "UT"))
        pa = float(row.get("PA", 0.0))
        ops = float(row.get("OPS", 0.0))
        speed = float(sprint_map.get(name, 27.0))
        base_on_base = sum(base_probs[key] for key in ("single", "double", "triple", "hr", "bb", "hbp"))
        sim_on_base = sum(sim_probs[key] for key in ("single", "double", "triple", "hr", "bb", "hbp"))
        base_xbh = sum(base_probs[key] for key in ("double", "triple", "hr"))
        sim_xbh = sum(sim_probs[key] for key in ("double", "triple", "hr"))
        delta_obp_pts = (sim_on_base - base_on_base) * 1000.0
        delta_hr_pts = (sim_probs["hr"] - base_probs["hr"]) * 1000.0

        if speed >= 29.0:
            archetype = "Speed"
        elif sim_probs["hr"] >= 0.050:
            archetype = "Power"
        elif sim_on_base >= 0.340:
            archetype = "Table Setter"
        else:
            archetype = "Balanced"

        player_rows.append(
            {
                "player": name,
                "pos": pos,
                "pa": pa,
                "ops": ops,
                "start_weight": float(row.get("start_w", 0.5)),
                "speed": speed,
                "archetype": archetype,
                "sim_on_base": sim_on_base,
                "sim_hr": sim_probs["hr"],
                "sim_xbh": sim_xbh,
                "sim_k": sim_probs["k"],
                "delta_obp_pts": delta_obp_pts,
                "delta_hr_pts": delta_hr_pts,
                "impact_score": pa * float(row.get("start_w", 0.5)) * sim_on_base,
                "boosted": bool(scenario_boosts and name in scenario_boosts),
            }
        )

    players = pd.DataFrame(player_rows)
    players = players.sort_values(
        ["boosted", "impact_score", "ops", "pa"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    return players


def _build_pitcher_projection_table(
    raw_dir: Path,
    starters: pd.DataFrame,
    bullpen: BullpenState,
    scenario_stats: dict[str, float],
) -> pd.DataFrame:
    baseline_team_era = max(0.1, bullpen.team_era_actual)
    era_scale = float(scenario_stats["ERA"]) / baseline_team_era
    pitcher_logs = pd.read_csv(raw_dir / "rangers_pitcher_gamelogs.csv")
    pitcher_logs = _coerce_numeric(pitcher_logs, ["GS", "ER", "H", "BB", "SO"])
    pitcher_logs["_ip"] = pitcher_logs["IP"].map(_ip_to_float)
    pitcher_logs["name"] = pitcher_logs["name"].map(_normalize_name)
    baseline_whip = float((pitcher_logs["H"].sum() + pitcher_logs["BB"].sum()) / max(0.1, pitcher_logs["_ip"].sum()))
    whip_scale = float(scenario_stats["WHIP"]) / max(0.1, baseline_whip)
    k_scale = float(scenario_stats["K9"]) / max(0.1, 8.5)

    starter_rows: list[dict[str, float | str]] = []
    for _, row in starters.iterrows():
        name = _normalize_name(row["Name"])
        sim_era = float(row["ERA"]) * era_scale
        sim_ip = float(row["avg_ip"])
        role = "SP1" if len(starter_rows) == 0 else "SP"
        starter_rows.append(
            {
                "player": name,
                "role": role,
                "usage": float(row["start_w"]),
                "sim_era": sim_era,
                "sim_whip": min(1.60, max(0.85, sim_era / 3.25 * 1.05 * whip_scale)),
                "sim_k9": min(12.5, max(5.0, float(scenario_stats["K9"]) * (float(row["ERA"]) / max(0.1, sim_era)) ** 0.15)),
                "sim_ip": sim_ip,
                "leverage": sim_ip * float(row["start_w"]),
                "delta_era": sim_era - float(row["ERA"]),
                "archetype": "Rotation Anchor" if role == "SP1" else "Starter",
            }
        )
    relievers = pitcher_logs[pitcher_logs["GS"] == 0].copy()
    reliever_stats = (
        relievers.groupby("name", as_index=False)
        .agg(
            G=("name", "size"),
            IP=("_ip", "sum"),
            ER=("ER", "sum"),
            H=("H", "sum"),
            BB=("BB", "sum"),
            K=("SO", "sum"),
        )
        .rename(columns={"name": "Name"})
    )
    reliever_stats["ERA"] = reliever_stats["ER"] * 9.0 / reliever_stats["IP"].clip(lower=0.1)
    reliever_stats["WHIP"] = (reliever_stats["H"] + reliever_stats["BB"]) / reliever_stats["IP"].clip(lower=0.1)
    reliever_stats["K9"] = reliever_stats["K"] * 9.0 / reliever_stats["IP"].clip(lower=0.1)
    reliever_stats["role"] = np.where(reliever_stats["Name"].isin(bullpen.closer_names), "Closer", "Setup")

    reliever_rows: list[dict[str, float | str]] = []
    for _, row in reliever_stats.sort_values(["role", "IP"], ascending=[True, False]).head(6).iterrows():
        name = _normalize_name(row["Name"])
        base_era = float(row["ERA"])
        sim_era = base_era * era_scale
        reliever_rows.append(
            {
                "player": name,
                "role": str(row["role"]),
                "usage": float(row["IP"]),
                "sim_era": sim_era,
                "sim_whip": min(1.70, max(0.85, float(row["WHIP"]) * whip_scale)),
                "sim_k9": min(13.5, max(5.5, float(row["K9"]) * k_scale / max(0.1, 8.5))),
                "sim_ip": float(row["IP"]),
                "leverage": float(row["IP"]) * (1.35 if str(row["role"]) == "Closer" else 1.0),
                "delta_era": sim_era - base_era,
                "archetype": "Closer" if str(row["role"]) == "Closer" else "Late Relief",
            }
        )

    pitchers = pd.DataFrame(starter_rows + reliever_rows)
    pitchers = pitchers.sort_values(
        ["role", "leverage", "sim_ip"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    return pitchers


def _build_bullpen_state(raw_dir: Path) -> BullpenState:
    pitcher_logs = pd.read_csv(raw_dir / "rangers_pitcher_gamelogs.csv")
    pitcher_logs = _coerce_numeric(pitcher_logs, ["GS", "ER", "H", "BB", "SO"])
    pitcher_logs["_ip"] = pitcher_logs["IP"].map(_ip_to_float)
    pitcher_logs["name"] = pitcher_logs["name"].map(_normalize_name)

    starters = pitcher_logs[pitcher_logs["GS"] == 1].copy()
    relievers = pitcher_logs[pitcher_logs["GS"] == 0].copy()

    team_era_actual = float(pitcher_logs["ER"].sum() * 9.0 / max(1.0, pitcher_logs["_ip"].sum()))
    starter_era_actual = float(starters["ER"].sum() * 9.0 / max(1.0, starters["_ip"].sum()))
    bullpen_era = float(relievers["ER"].sum() * 9.0 / max(1.0, relievers["_ip"].sum()))
    starter_ip_mean = float(starters["_ip"].mean())
    starter_ip_std = float(starters["_ip"].std(ddof=0)) if len(starters) > 1 else 0.75

    save_df = pd.read_csv(raw_dir / "tex_2025_save_situation_splits.csv")
    save_df = _coerce_numeric(save_df, ["GF", "SV", "IP", "ER"])
    save_df["Name"] = save_df["Name"].map(_normalize_name)
    closer_names = set(
        save_df[
            (save_df["Name"] != "Team Total") &
            ((save_df["GF"].fillna(0) >= 5) | (save_df["SV"].fillna(0) >= 3))
        ]["Name"].tolist()
    )
    all_save_names = set(save_df["Name"].tolist())
    # Mar-Apr: Jackson이 주 마무리
    closer_pool_p1 = [n for n in ["Luke Jackson"] if n in closer_names]
    # May-Jul: Garcia 주, Armstrong 보조 (순서가 가중치 역할)
    closer_pool_p2 = [n for n in ["Robert Garcia", "Shawn Armstrong"]
                      if n in all_save_names]
    # Aug-Sep: Armstrong 주, Garcia 보조 (실제 데이터 기반 교체 시점)
    closer_pool_p3 = [n for n in ["Shawn Armstrong", "Robert Garcia"]
                      if n in all_save_names]
    if not closer_pool_p1:
        closer_pool_p1 = list(closer_names)
    if not closer_pool_p2:
        closer_pool_p2 = list(closer_names)
    if not closer_pool_p3:
        closer_pool_p3 = list(closer_names)

    reliever_stats = (
        relievers.groupby("name", as_index=False)
        .agg(
            G=("name", "size"),
            IP=("_ip", "sum"),
            ER=("ER", "sum"),
            H=("H", "sum"),
            BB=("BB", "sum"),
            K=("SO", "sum"),
        )
        .rename(columns={"name": "Name"})
    )
    reliever_stats["ERA"] = reliever_stats["ER"] * 9.0 / reliever_stats["IP"].clip(lower=0.1)
    reliever_stats["role"] = np.where(reliever_stats["Name"].isin(closer_names), "closer", "setup")
    reliever_stats["usage_weight"] = np.where(
        reliever_stats["role"] == "closer",
        reliever_stats["G"].clip(lower=1),
        reliever_stats["IP"].clip(lower=1.0),
    )

    setup_df = reliever_stats[~reliever_stats["Name"].isin(closer_names)].copy()
    if setup_df.empty:
        setup_df = reliever_stats.copy()
    closer_df = reliever_stats[reliever_stats["Name"].isin(closer_names)].copy()
    if closer_df.empty:
        closer_df = reliever_stats.copy()

    setup_era = float((setup_df["ER"].sum() * 9.0) / max(0.1, setup_df["IP"].sum()))
    closer_era = float((closer_df["ER"].sum() * 9.0) / max(0.1, closer_df["IP"].sum()))

    inning_splits = pd.read_csv(raw_dir / "tex_2025_pitching_inning_splits.csv")
    inning_splits = _coerce_numeric(inning_splits, ["ERA"])
    inning_map = {}
    for _, row in inning_splits.iterrows():
        split = str(row["Split"])
        if split.startswith("1st"):
            inning_map[1] = float(row["ERA"])
        elif split.startswith("2nd"):
            inning_map[2] = float(row["ERA"])
        elif split.startswith("3rd"):
            inning_map[3] = float(row["ERA"])
        elif split.startswith("4th"):
            inning_map[4] = float(row["ERA"])
        elif split.startswith("5th"):
            inning_map[5] = float(row["ERA"])
        elif split.startswith("6th"):
            inning_map[6] = float(row["ERA"])
        elif split.startswith("7th"):
            inning_map[7] = float(row["ERA"])
        elif split.startswith("8th"):
            inning_map[8] = float(row["ERA"])
        elif split.startswith("9th"):
            inning_map[9] = float(row["ERA"])

    return BullpenState(
        inning_era_map=inning_map,
        team_era_actual=team_era_actual,
        starter_era_actual=starter_era_actual,
        starter_ip_mean=starter_ip_mean,
        starter_ip_std=max(0.25, starter_ip_std),
        closer_era=closer_era,
        setup_era=setup_era,
        bullpen_era=bullpen_era,
        reliever_stats=reliever_stats,
        closer_names=closer_names,
        closer_pool_p1=closer_pool_p1,
        closer_pool_p2=closer_pool_p2,
        closer_pool_p3=closer_pool_p3,
    )


def _build_park_factor_map(raw_dir: Path) -> dict[str, dict[str, float]]:
    parks = pd.read_csv(raw_dir / "fangraphs-guts-data.csv")
    parks = _coerce_numeric(parks, ["Basic (5yr)", "1B", "2B", "3B", "HR", "BB"])
    mapping: dict[str, dict[str, float]] = {}
    for _, row in parks.iterrows():
        team = str(row["Team"]).strip()
        mapping[team] = {
            "single": float(row["1B"]) / 100.0,
            "double": float(row["2B"]) / 100.0,
            "triple": float(row["3B"]) / 100.0,
            "hr": float(row["HR"]) / 100.0,
            "bb": float(row["BB"]) / 100.0,
            "overall": float(row["Basic (5yr)"]) / 100.0,
        }
    return mapping


def _build_opponent_strength(raw_dir: Path) -> dict[str, float]:
    team_hist = pd.read_csv(raw_dir / "mlb_team_seasons.csv")
    team_hist = _coerce_numeric(team_hist, ["year", "RS", "G"])
    team_hist = team_hist[team_hist["year"] == SEASON].copy()
    team_hist["rspg"] = team_hist["RS"] / team_hist["G"].clip(lower=1.0)
    league_rspg = float(team_hist["rspg"].mean())
    full_strength: dict[str, float] = {}
    for _, row in team_hist.iterrows():
        team_name = str(row["team"])
        ratio = float(row["rspg"]) / max(0.1, league_rspg)
        full_strength[team_name] = min(1.20, max(0.82, ratio))
    strength: dict[str, float] = {}
    for code, short in TEAM_CODE_TO_NAME.items():
        for full_name, val in full_strength.items():
            if full_name.endswith(short):
                strength[code] = val
                break
    return strength


def _build_opponent_pitch_factors(raw_dir: Path) -> dict[str, dict[str, float]]:
    teams = pd.read_csv(raw_dir / "mlb_teams_2025_pitching.csv")
    teams = _coerce_numeric(teams, ["ERA", "K/9", "BB/9", "HR/9"])
    lg_k9 = float(teams["K/9"].mean())
    lg_bb9 = float(teams["BB/9"].mean())
    lg_hr9 = float(teams["HR/9"].mean())
    lg_era = float(teams["ERA"].mean())
    factors: dict[str, dict[str, float]] = {}
    for _, row in teams.iterrows():
        code = str(row["Team"]).strip()
        entry = {
            "k_factor": float(np.clip(float(row["K/9"]) / max(0.1, lg_k9), 0.85, 1.15)),
            "bb_factor": float(np.clip(float(row["BB/9"]) / max(0.1, lg_bb9), 0.85, 1.15)),
            "hr_factor": float(np.clip(float(row["HR/9"]) / max(0.1, lg_hr9), 0.85, 1.15)),
            "hit_factor": float(np.clip(lg_era / max(0.1, float(row["ERA"])), 0.85, 1.15)),
        }
        factors[code] = entry
        long_name = TEAM_CODE_TO_NAME.get(code)
        if long_name:
            factors[long_name.split()[-1]] = entry
    return factors


def _build_schedule_context(schedule_df: pd.DataFrame, opponent_strength: dict[str, float]) -> pd.DataFrame:
    frame = schedule_df.copy()
    frame["month"] = frame["month"].fillna("UNK")
    frame["opp_strength"] = frame["Opp"].map(opponent_strength).fillna(1.0)
    summary = (
        frame.groupby("month", sort=False)
        .agg(
            games=("Opp", "size"),
            home_games=("is_home", "sum"),
            avg_opp_strength=("opp_strength", "mean"),
        )
        .reset_index()
    )
    summary["away_games"] = summary["games"] - summary["home_games"]
    summary["strength_index"] = (summary["avg_opp_strength"] * 100).round(1)
    summary["difficulty"] = np.where(
        summary["strength_index"] >= 103,
        "Tough",
        np.where(summary["strength_index"] <= 97, "Soft", "Neutral"),
    )
    return summary[["month", "games", "home_games", "away_games", "strength_index", "difficulty"]]


def _build_series_context(schedule_df: pd.DataFrame, opponent_strength: dict[str, float]) -> pd.DataFrame:
    frame = schedule_df.copy()
    frame["opp_strength"] = frame["Opp"].map(opponent_strength).fillna(1.0)
    grouped = (
        frame.groupby("series_id", sort=True)
        .agg(
            start_date=("date", "min"),
            end_date=("date", "max"),
            month=("month", "first"),
            opponent=("Opp", "first"),
            is_home=("is_home", "first"),
            games=("Opp", "size"),
            strength_index=("opp_strength", "mean"),
        )
        .reset_index(drop=True)
    )
    grouped["venue"] = np.where(grouped["is_home"], "Home", "Away")
    grouped["strength_index"] = (grouped["strength_index"] * 100).round(1)
    grouped["difficulty"] = np.where(
        grouped["strength_index"] >= 103,
        "Tough",
        np.where(grouped["strength_index"] <= 97, "Soft", "Neutral"),
    )
    grouped["series_label"] = grouped.apply(
        lambda row: f'{row["month"]} {row["start_date"].day}-{row["end_date"].day} vs {row["opponent"]}'
        if row["is_home"]
        else f'{row["month"]} {row["start_date"].day}-{row["end_date"].day} at {row["opponent"]}',
        axis=1,
    )
    return grouped[
        ["series_label", "month", "opponent", "venue", "games", "strength_index", "difficulty"]
    ]


def _simulate_schedule(
    schedule_df: pd.DataFrame,
    lineup_pool: LineupPool,
    starters: pd.DataFrame,
    bullpen: BullpenState,
    advancement: AdvancementTables,
    scenario_stats: dict[str, float],
    opponent_strength: dict[str, float],
    park_factors: dict[str, float],
    opponent_pitch_factors: dict[str, dict[str, float]],
    n_sims: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    weights = starters["start_w"].to_numpy(dtype=float)
    weights = weights / weights.sum()
    wins = []
    month_order = list(pd.Index(schedule_df["month"].fillna("UNK")).drop_duplicates())
    month_index = {month: idx for idx, month in enumerate(month_order)}
    monthly_wins = np.zeros((n_sims, len(month_order)), dtype=float)
    series_order = list(pd.Index(schedule_df["series_id"]).drop_duplicates())
    series_index = {series_id: idx for idx, series_id in enumerate(series_order)}
    series_wins = np.zeros((n_sims, len(series_order)), dtype=float)

    for sim_idx in range(n_sims):
        season_wins = 0.0
        bullpen_tracker: dict[str, dict[str, float]] = {}
        recent_apps: dict[str, list[int]] = {}
        for _, game in schedule_df.iterrows():
            lineup_probs, profiles, _ = lineup_pool.sample()
            opp_code = str(game["Opp"]).strip()
            if bool(game.get("is_home", False)):
                stadium_key = "Rangers"
            else:
                stadium_key = TEAM_CODE_TO_NAME.get(opp_code, opp_code).split()[-1]
            park = park_factors.get(stadium_key, {"overall": 1.0})
            pitch_adj = opponent_pitch_factors.get(
                opp_code,
                {"k_factor": 1.0, "bb_factor": 1.0, "hr_factor": 1.0, "hit_factor": 1.0},
            )

            adjusted_probs = []
            for probs in lineup_probs:
                p1 = _scale_probs_for_park(probs, park)
                p2 = _adjust_probs_for_pitcher(p1, **pitch_adj)
                adjusted_probs.append(p2)

            starter = starters.iloc[int(np.random.choice(len(starters), p=weights))]
            park_factor = park.get("overall", 1.0)
            is_home = bool(game.get("is_home", False))

            if is_home:
                tex_8 = 0
                lineup_idx = 0
                for _ in range(8):
                    inning_runs, lineup_idx = _simulate_inning(adjusted_probs, profiles, lineup_idx, advancement=advancement)
                    tex_8 += inning_runs
                opp_runs = _simulate_opponent_runs(
                    bullpen=bullpen,
                    scenario_era=float(scenario_stats["ERA"]),
                    starter_era=float(starter["ERA"]),
                    starter_ip=float(starter["avg_ip"]),
                    park_factor=park_factor,
                    opp_ratio=opponent_strength.get(opp_code, 1.0),
                    game_index=int(game.name) + 1,
                    tracker=bullpen_tracker,
                    recent_apps=recent_apps,
                    texas_runs_hint=tex_8,
                )
                if tex_8 > opp_runs:
                    texas_runs = tex_8
                else:
                    ninth_runs, _ = _simulate_inning(adjusted_probs, profiles, lineup_idx, advancement=advancement)
                    texas_runs = tex_8 + ninth_runs
            else:
                texas_runs = _simulate_texas_runs(adjusted_probs, profiles, advancement=advancement)
                opp_runs = _simulate_opponent_runs(
                    bullpen=bullpen,
                    scenario_era=float(scenario_stats["ERA"]),
                    starter_era=float(starter["ERA"]),
                    starter_ip=float(starter["avg_ip"]),
                    park_factor=park_factor,
                    opp_ratio=opponent_strength.get(opp_code, 1.0),
                    game_index=int(game.name) + 1,
                    tracker=bullpen_tracker,
                    recent_apps=recent_apps,
                    texas_runs_hint=texas_runs,
                )

            if texas_runs > opp_runs:
                season_wins += 1.0
                game_win = 1.0
            elif texas_runs == opp_runs:
                game_win = 1.0 if np.random.random() < float(scenario_stats["xi_wp"]) else 0.0
                season_wins += game_win
            else:
                game_win = 0.0
            monthly_wins[sim_idx, month_index[str(game.get("month", "UNK"))]] += game_win
            series_wins[sim_idx, series_index[int(game["series_id"])]] += game_win
        wins.append(season_wins)

    monthly_summary = pd.DataFrame(
        {
            "month": month_order,
            "mean_wins": monthly_wins.mean(axis=0).round(2),
            "p25_wins": np.quantile(monthly_wins, 0.25, axis=0).round(2),
            "p75_wins": np.quantile(monthly_wins, 0.75, axis=0).round(2),
        }
    )
    monthly_summary["cumulative_mean"] = monthly_summary["mean_wins"].cumsum().round(2)
    series_summary = pd.DataFrame(
        {
            "series_id": series_order,
            "mean_wins": series_wins.mean(axis=0).round(2),
            "p25_wins": np.quantile(series_wins, 0.25, axis=0).round(2),
            "p75_wins": np.quantile(series_wins, 0.75, axis=0).round(2),
        }
    )
    games_per_series = schedule_df.groupby("series_id").size().reindex(series_order).to_numpy(dtype=float)
    series_summary["series_win_prob"] = (
        (series_wins > (games_per_series / 2.0)).mean(axis=0).round(3)
    )
    return np.array(wins, dtype=float), monthly_summary, series_summary


def _calibration_offset(bundle: ModelBundle) -> float:
    baseline = bundle.tex25["pyth_W"] + _predict_residual(bundle, bundle.tex25)
    return float(bundle.tex25["W"] - baseline)


def run_simulation(
    raw_dir: str | Path,
    scenario_name: str,
    n_sims: int = 300,
    custom_stats: dict | None = None,
    custom_boosts: dict | None = None,
    fast_mode: bool = True,
) -> dict[str, object]:
    raw_path = Path(raw_dir)

    use_cache = fast_mode and custom_stats is None and custom_boosts is None
    if use_cache:
        cache_file = _simulation_cache_file(raw_path, scenario_name, n_sims)
        if cache_file.exists():
            try:
                with cache_file.open("rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, dict) and "distribution" in cached and "summary" in cached:
                    return cached
            except Exception:
                pass

    bundle = _train_model_bundle(raw_path)
    scenarios = _build_scenarios(bundle.tex25)
    scenario = scenarios[scenario_name]

    effective_stats = dict(scenario["stats"])
    if custom_stats:
        effective_stats.update(custom_stats)

    effective_boosts = custom_boosts if custom_boosts is not None else scenario["boosts"]

    schedule_df = _clean_schedule(pd.read_csv(raw_path / "texas_2025_game_log.csv"))
    batting_pool, sprint_map, hbp_df = _build_batting_pool(raw_path)
    starters = _build_starters(raw_path)
    bullpen = _build_bullpen_state(raw_path)
    advancement = _build_advancement_tables(raw_path)
    lineup_pool = LineupPool(batting_pool, sprint_map, hbp_df, effective_boosts)
    player_projection = _build_player_projection_table(
        batting_pool=batting_pool,
        sprint_map=sprint_map,
        hbp_df=hbp_df,
        scenario_boosts=effective_boosts,
    )
    pitcher_projection = _build_pitcher_projection_table(
        raw_dir=raw_path,
        starters=starters,
        bullpen=bullpen,
        scenario_stats=effective_stats,
    )
    opponent_strength = _build_opponent_strength(raw_path)
    park_factors = _build_park_factor_map(raw_path)
    opponent_pitch_factors = _build_opponent_pitch_factors(raw_path)
    schedule_context = _build_schedule_context(schedule_df, opponent_strength)
    series_context = _build_series_context(schedule_df, opponent_strength)

    raw_distribution, monthly_summary, series_summary = _simulate_schedule(
        schedule_df=schedule_df,
        lineup_pool=lineup_pool,
        starters=starters,
        bullpen=bullpen,
        advancement=advancement,
        scenario_stats=effective_stats,
        opponent_strength=opponent_strength,
        park_factors=park_factors,
        opponent_pitch_factors=opponent_pitch_factors,
        n_sims=n_sims,
    )

    residual_bonus = _predict_residual(bundle, effective_stats)
    calibration_offset = _calibration_offset(bundle)
    noise = np.random.normal(0.0, bundle.ensemble_cv_mae, size=n_sims)
    adjusted = np.clip(raw_distribution + residual_bonus + calibration_offset + noise, 55.0, 110.0)
    distribution = pd.DataFrame({"wins": adjusted.round(2)})

    result = {
        "distribution": distribution,
        "summary": {
            "mean": float(distribution["wins"].mean()),
            "median": float(distribution["wins"].median()),
            "p10": float(distribution["wins"].quantile(0.10)),
            "p90": float(distribution["wins"].quantile(0.90)),
            "over_81_5": float((distribution["wins"] >= 82).mean()),
            "over_87_5": float((distribution["wins"] >= 88).mean()),
            "residual_bonus": float(residual_bonus),
            "calibration_offset": float(calibration_offset),
            "ensemble_cv_mae": float(bundle.ensemble_cv_mae),
        },
        "scenarios": list(scenarios.keys()),
        "scenario_stats": effective_stats,
        "monthly_summary": monthly_summary,
        "schedule_context": schedule_context,
        "series_summary": series_context.join(series_summary.drop(columns=["series_id"])),
        "player_projection": player_projection,
        "pitcher_projection": pitcher_projection,
    }

    if use_cache:
        cache_file = _simulation_cache_file(raw_path, scenario_name, n_sims)
        with cache_file.open("wb") as handle:
            pickle.dump(result, handle)

    return result


def build_scenario_snapshots(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    raw_path = Path(raw_dir)
    bundle = _train_model_bundle(raw_path)
    scenarios = _build_scenarios(bundle.tex25)
    batting_pool, sprint_map, hbp_df = _build_batting_pool(raw_path)
    starters = _build_starters(raw_path)
    bullpen = _build_bullpen_state(raw_path)

    hitter_frames: list[pd.DataFrame] = []
    pitcher_frames: list[pd.DataFrame] = []
    for scenario_name, scenario in scenarios.items():
        hitter_frame = _build_player_projection_table(
            batting_pool=batting_pool,
            sprint_map=sprint_map,
            hbp_df=hbp_df,
            scenario_boosts=scenario["boosts"],
        ).copy()
        hitter_frame["scenario"] = scenario_name
        hitter_frames.append(hitter_frame)

        pitcher_frame = _build_pitcher_projection_table(
            raw_dir=raw_path,
            starters=starters,
            bullpen=bullpen,
            scenario_stats=scenario["stats"],
        ).copy()
        pitcher_frame["scenario"] = scenario_name
        pitcher_frames.append(pitcher_frame)

    return {
        "hitters": pd.concat(hitter_frames, ignore_index=True),
        "pitchers": pd.concat(pitcher_frames, ignore_index=True),
    }


def get_batter_options(raw_dir: str | Path) -> list[str]:
    pool, _, _ = _build_batting_pool(Path(raw_dir))
    return sorted(pool["Name"].tolist())


def get_scenario_defaults(raw_dir: str | Path) -> dict:
    bundle = _train_model_bundle(Path(raw_dir))
    scenarios = _build_scenarios(bundle.tex25)
    return {
        "tex25": bundle.tex25,
        "scenarios": scenarios,
    }
