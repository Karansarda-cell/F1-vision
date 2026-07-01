"""
Monaco 2026 Counterfactual Standings Simulator
-----------------------------------------------
Answers: "How would standings look if specific incidents didn't happen?"

USAGE:
  python monaco_counterfactual.py

REQUIRES: pip install fastf1 pandas numpy tabulate matplotlib

HOW IT WORKS:
  1. Loads actual race lap data up to the incident lap
  2. Identifies each driver's representative pace (3 modes)
  3. Projects remaining laps with tyre degradation
  4. Recomputes gaps and produces simulated final standings
  5. Compares simulated vs actual finishing order

CONFIG: Edit the CONFIG block below to switch year/round/incidents
"""

import os, warnings, logging
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
YEAR        = 2026          # Change to 2025 if 2026 not indexed yet
RACE_NAME   = "Monaco"
CACHE_DIR   = ".fastf1-cache"

# Incident lap — red flag / retirement cutoff
# 2026 Monaco: red flag lap 68 of 78 (Leclerc crash, track surface)
# 2025 Monaco: full green flag race, so set to 78 to run full sim
INCIDENT_LAP = 68
TOTAL_LAPS   = 78

# Drivers to REMOVE from counterfactual (DNFs before incident)
# 2026: Verstappen DNF lap 1 (power unit)
REMOVE_DRIVERS = ["VER"]

# Tyre degradation per lap (seconds added per lap on same tyre)
# Monaco is very low deg — approx 0.02s/lap
TYRE_DEG_PER_LAP = 0.02

# Pace projection modes
PACE_MODES = {
    "conservative": "Average pace from current stint",
    "optimistic":   "Fastest clean-air lap of the race",
    "realistic":    "Median of last 5 laps before incident"
}

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*58}")
print(f"  Monaco {YEAR} Counterfactual Standings Simulator")
print(f"{'='*58}\n")
print(f"Loading {YEAR} {RACE_NAME} GP race data...")

session = fastf1.get_session(YEAR, RACE_NAME, 'R')
session.load(telemetry=False, laps=True, weather=False)

laps = session.laps.copy()
laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

# Get actual results for comparison
results = session.results[["Abbreviation", "FullName", "TeamName",
                            "Position", "Status", "Points"]].copy()
results = results.sort_values("Position")

print(f"  Drivers loaded  : {len(laps['Driver'].unique())}")
print(f"  Total laps      : {len(laps)}")
print(f"  Incident lap    : {INCIDENT_LAP} of {TOTAL_LAPS}")
print(f"  Removed (DNF)   : {REMOVE_DRIVERS}")

# ─────────────────────────────────────────────────────────────
# 2. COMPUTE POSITIONS AT INCIDENT LAP
# ─────────────────────────────────────────────────────────────
pre_incident = laps[laps["LapNumber"] <= INCIDENT_LAP].copy()
pre_incident = pre_incident[~pre_incident["Driver"].isin(REMOVE_DRIVERS)]

# Filter out SC/VSC laps and pit laps for pace computation
clean_laps = pre_incident[
    (pre_incident["LapTimeSec"].notna()) &
    (pre_incident["LapTimeSec"] > 60) &       # Monaco min ~65s
    (pre_incident["LapTimeSec"] < 120) &      # filter obvious outliers/SC
    (pre_incident["PitInTime"].isna()) &
    (pre_incident["PitOutTime"].isna())
].copy()

# ─────────────────────────────────────────────────────────────
# 3. COMPUTE CUMULATIVE RACE TIME AT INCIDENT LAP
# ─────────────────────────────────────────────────────────────
# Sum actual lap times up to incident lap per driver
cum_time = (
    pre_incident.groupby("Driver")["LapTimeSec"]
    .sum()
    .reset_index()
    .rename(columns={"LapTimeSec": "CumTimeSec"})
)

# Laps completed per driver at incident
laps_done = (
    pre_incident.groupby("Driver")["LapNumber"]
    .max()
    .reset_index()
    .rename(columns={"LapNumber": "LapsCompleted"})
)

driver_state = cum_time.merge(laps_done, on="Driver")

# ─────────────────────────────────────────────────────────────
# 4. COMPUTE PACE PER DRIVER (3 modes)
# ─────────────────────────────────────────────────────────────
def get_pace(driver_code, mode):
    d = clean_laps[clean_laps["Driver"] == driver_code]["LapTimeSec"]
    if len(d) == 0:
        # fallback: median of all drivers
        return clean_laps["LapTimeSec"].median()
    if mode == "conservative":
        return d.mean()
    elif mode == "optimistic":
        return d.min()
    elif mode == "realistic":
        last5 = clean_laps[
            (clean_laps["Driver"] == driver_code) &
            (clean_laps["LapNumber"] >= INCIDENT_LAP - 5)
        ]["LapTimeSec"]
        return last5.median() if len(last5) > 0 else d.median()

# ─────────────────────────────────────────────────────────────
# 5. PROJECT REMAINING LAPS + COMPUTE SIMULATED STANDINGS
# ─────────────────────────────────────────────────────────────
def simulate_standings(mode):
    remaining_laps = TOTAL_LAPS - INCIDENT_LAP
    results_sim = []

    for _, row in driver_state.iterrows():
        driver = row["Driver"]
        cum_time_at_incident = row["CumTimeSec"]
        laps_completed = row["LapsCompleted"]

        base_pace = get_pace(driver, mode)
        remaining = TOTAL_LAPS - laps_completed

        # Project remaining laps with tyre degradation
        projected_time = sum(
            base_pace + (TYRE_DEG_PER_LAP * i)
            for i in range(remaining)
        )

        total_time = cum_time_at_incident + projected_time

        results_sim.append({
            "Driver": driver,
            "LapsCompleted": laps_completed,
            "PaceUsed_s": round(base_pace, 3),
            "ProjectedRemaining_s": round(projected_time, 2),
            "TotalRaceTime_s": round(total_time, 2),
        })

    sim_df = pd.DataFrame(results_sim).sort_values("TotalRaceTime_s").reset_index(drop=True)
    sim_df.index += 1
    sim_df["SimPosition"] = sim_df.index

    # Format total time as mm:ss.sss
    def fmt_time(s):
        m = int(s // 60)
        sec = s % 60
        return f"{m}:{sec:06.3f}"

    sim_df["TotalRaceTime"] = sim_df["TotalRaceTime_s"].apply(fmt_time)

    # Gap to leader
    leader_time = sim_df.iloc[0]["TotalRaceTime_s"]
    sim_df["GapToLeader"] = sim_df["TotalRaceTime_s"].apply(
        lambda x: f"+{x - leader_time:.3f}s" if x != leader_time else "LEADER"
    )

    return sim_df

# ─────────────────────────────────────────────────────────────
# 6. PRINT RESULTS FOR ALL 3 MODES
# ─────────────────────────────────────────────────────────────
all_sims = {}

for mode, desc in PACE_MODES.items():
    print(f"\n{'─'*58}")
    print(f"  MODE: {mode.upper()} — {desc}")
    print(f"{'─'*58}")

    sim = simulate_standings(mode)
    all_sims[mode] = sim

    display_cols = ["SimPosition", "Driver", "PaceUsed_s", "TotalRaceTime", "GapToLeader"]
    print(sim[display_cols].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 7. DELTA TABLE — actual vs simulated position per driver
# ─────────────────────────────────────────────────────────────
print(f"\n{'─'*58}")
print(f"  POSITION DELTA: Actual vs Counterfactual (realistic mode)")
print(f"{'─'*58}")

actual_pos = results[["Abbreviation", "Position", "FullName"]].rename(
    columns={"Abbreviation": "Driver", "Position": "ActualPos"}
)
actual_pos["ActualPos"] = pd.to_numeric(actual_pos["ActualPos"], errors="coerce")

delta = all_sims["realistic"][["Driver", "SimPosition"]].merge(
    actual_pos, on="Driver", how="left"
)
delta["Delta"] = delta["ActualPos"] - delta["SimPosition"]
delta["Change"] = delta["Delta"].apply(
    lambda x: f"▲ {int(x)}" if x > 0 else (f"▼ {int(abs(x))}" if x < 0 else "━ same")
)

print(delta[["SimPosition", "Driver", "FullName", "ActualPos", "Change"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# 8. PLOT — Simulated standings comparison across modes
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 8))
fig.suptitle(
    f"Monaco {YEAR} GP — Counterfactual Standings\n"
    f"(If incident at lap {INCIDENT_LAP} did not happen)",
    fontsize=13, fontweight="bold", y=1.01
)

colors = plt.cm.tab20.colors

for ax, (mode, desc) in zip(axes, PACE_MODES.items()):
    sim = all_sims[mode]
    drivers = sim["Driver"].tolist()
    gaps = [0.0] + [
        float(g.replace("+", "").replace("s", ""))
        for g in sim["GapToLeader"].tolist()[1:]
    ]

    bar_colors = [colors[i % len(colors)] for i in range(len(drivers))]
    bars = ax.barh(range(len(drivers)), gaps, color=bar_colors, edgecolor="white", linewidth=0.5)

    ax.set_yticks(range(len(drivers)))
    ax.set_yticklabels([f"P{i+1}  {d}" for i, d in enumerate(drivers)], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Gap to Leader (s)")
    ax.set_title(f"{mode.capitalize()}\n{desc}", fontsize=9)
    ax.axvline(0, color="white", linewidth=0.5)
    ax.grid(axis="x", alpha=0.2)

    for bar, gap in zip(bars, gaps):
        if gap > 0:
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                    f"+{gap:.1f}s", va="center", fontsize=7, alpha=0.8)

plt.tight_layout()
plt.savefig("monaco_counterfactual_standings.png", dpi=150, bbox_inches="tight",
            facecolor="#1a1a2e")
print(f"\n✓ Plot saved: monaco_counterfactual_standings.png")

# Export CSV
all_sims["realistic"].to_csv("counterfactual_standings_realistic.csv", index=False)
print(f"✓ CSV saved: counterfactual_standings_realistic.csv")
print(f"\nDone.\n")
