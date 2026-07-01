"""
Monaco F1 Race Analysis Pipeline
Uses FastF1 to pull lap times, sector times, pit stops, and corner telemetry.
"""

import os
import warnings
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")


YEAR      = 2025
RACE_NAME = "Monaco"
SESSION   = "R"          # R=Race, Q=Qualifying, FP1/FP2/FP3
CACHE_DIR = "./f1_cache"

# Drivers to focus on (use 3-letter codes). None = all drivers.
FOCUS_DRIVERS = ["NOR", "LEC", "PIA","VER","HAM"]   # 2025 podium

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False)


# 1. LOAD SESSION

print(f"\n{'='*55}")
print(f"  Loading {YEAR} {RACE_NAME} GP — {SESSION}")
print(f"{'='*55}\n")

session = fastf1.get_session(YEAR, RACE_NAME, SESSION)
session.load(telemetry=True, laps=True, weather=True)

print(f"Event    : {session.event['EventName']}")
print(f"Date     : {session.event['EventDate'].date()}")
print(f"Circuit  : {session.event['Location']}")
print(f"Drivers  : {len(session.drivers)} loaded")


# 2. LAP TIMES — all drivers, all laps

print("\n--- LAP TIMES OVERVIEW ---")

laps = session.laps.copy()
laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()

# Fastest lap per driver
fastest_per_driver = (
    laps[laps["IsPersonalBest"] == True]
    [["Driver", "LapNumber", "LapTimeSec", "Compound"]]
    .sort_values("LapTimeSec")
    .reset_index(drop=True)
)
fastest_per_driver.index += 1
print(fastest_per_driver.to_string())


# 3. SECTOR TIMES — S1, S2, S3 per lap

print("\n--- SECTOR TIMES (focused drivers) ---")

sector_cols = ["Driver", "LapNumber", "Sector1Time", "Sector2Time",
               "Sector3Time", "LapTime", "Compound", "TyreLife"]

focus_laps = laps[laps["Driver"].isin(FOCUS_DRIVERS)][sector_cols].copy()
for col in ["Sector1Time", "Sector2Time", "Sector3Time", "LapTime"]:
    focus_laps[col.replace("Time", "Sec")] = focus_laps[col].dt.total_seconds()

focus_laps_clean = focus_laps.dropna(subset=["Sector1Sec", "Sector2Sec", "Sector3Sec"])
print(focus_laps_clean[["Driver", "LapNumber", "Sector1Sec",
                          "Sector2Sec", "Sector3Sec", "LapSec",
                          "Compound", "TyreLife"]].to_string(index=False))


# 4. PIT STOP ANALYSIS

print("\n--- PIT STOPS ---")

pit_laps = laps[laps["PitOutTime"].notna() | laps["PitInTime"].notna()].copy()
pit_laps["PitStopDurationSec"] = pit_laps["PitInTime"].dt.total_seconds()

pit_summary = (
    laps[laps["Driver"].isin(FOCUS_DRIVERS)]
    [["Driver", "LapNumber", "PitInTime", "PitOutTime", "Compound", "TyreLife"]]
    .dropna(subset=["PitInTime"])
    .reset_index(drop=True)
)
print(pit_summary.to_string(index=False))


# 5. TELEMETRY — corner speed analysis (fastest lap per driver)

print("\n--- TELEMETRY: CORNER SPEED (fastest lap) ---")

telemetry_data = {}

for driver in FOCUS_DRIVERS:
    try:
        fast_lap = session.laps.pick_driver(driver).pick_fastest()
        tel = fast_lap.get_telemetry().add_distance()
        tel["Driver"] = driver
        telemetry_data[driver] = tel

        # Identify corner entry (local speed minima = corner apex zones)
        # Rolling minimum over 200m windows as proxy for corner speed
        tel["SpeedSmooth"] = tel["Speed"].rolling(10, center=True).mean()
        local_min_mask = (
            (tel["SpeedSmooth"] < tel["SpeedSmooth"].shift(5)) &
            (tel["SpeedSmooth"] < tel["SpeedSmooth"].shift(-5)) &
            (tel["SpeedSmooth"] < tel["SpeedSmooth"].quantile(0.45))
        )
        corners = tel[local_min_mask][["Distance", "Speed", "Throttle", "Brake", "nGear"]]
        print(f"\n{driver} — {len(corners)} corner zones detected on fastest lap:")
        print(corners.to_string(index=False))

    except Exception as e:
        print(f"{driver}: telemetry load failed — {e}")


# 6. PLOT 1 — Lap time evolution per driver

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(f"{YEAR} Monaco GP — Race Analysis", fontsize=14, fontweight="bold")

ax1 = axes[0, 0]
for driver in FOCUS_DRIVERS:
    d_laps = laps[laps["Driver"] == driver].dropna(subset=["LapTimeSec"])
    d_laps = d_laps[d_laps["LapTimeSec"] < d_laps["LapTimeSec"].quantile(0.97)]  # remove SC laps
    color = fastf1.plotting.get_driver_color(driver, session)
    ax1.plot(d_laps["LapNumber"], d_laps["LapTimeSec"],
             label=driver, color=color, linewidth=1.5, alpha=0.85)

ax1.set_xlabel("Lap Number")
ax1.set_ylabel("Lap Time (s)")
ax1.set_title("Lap Time Evolution")
ax1.legend()
ax1.grid(alpha=0.3)


# PLOT 2 — Sector time comparison (box plot)

ax2 = axes[0, 1]
sector_plot_data = []
for driver in FOCUS_DRIVERS:
    d = focus_laps_clean[focus_laps_clean["Driver"] == driver]
    for _, row in d.iterrows():
        for sec, col in [("S1", "Sector1Sec"), ("S2", "Sector2Sec"), ("S3", "Sector3Sec")]:
            if pd.notna(row[col]):
                sector_plot_data.append({"Driver": driver, "Sector": sec, "Time": row[col]})

sector_df = pd.DataFrame(sector_plot_data)
for i, sec in enumerate(["S1", "S2", "S3"]):
    for j, driver in enumerate(FOCUS_DRIVERS):
        subset = sector_df[(sector_df["Sector"] == sec) & (sector_df["Driver"] == driver)]["Time"]
        x_pos = i * (len(FOCUS_DRIVERS) + 1) + j
        color = fastf1.plotting.get_driver_color(driver, session)
        ax2.boxplot(subset.dropna(), positions=[x_pos], widths=0.6,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, alpha=0.7),
                    medianprops=dict(color="white", linewidth=2))

ax2.set_title("Sector Time Distribution")
ax2.set_ylabel("Time (s)")
xtick_pos = [i * (len(FOCUS_DRIVERS) + 1) + len(FOCUS_DRIVERS) // 2 for i in range(3)]
ax2.set_xticks(xtick_pos)
ax2.set_xticklabels(["Sector 1", "Sector 2", "Sector 3"])
ax2.grid(alpha=0.3, axis="y")


# PLOT 3 — Speed trace overlay (fastest lap telemetry)

ax3 = axes[1, 0]
for driver, tel in telemetry_data.items():
    color = fastf1.plotting.get_driver_color(driver, session)
    ax3.plot(tel["Distance"], tel["Speed"],
             label=driver, color=color, linewidth=1.2, alpha=0.85)

ax3.set_xlabel("Distance (m)")
ax3.set_ylabel("Speed (km/h)")
ax3.set_title("Speed Trace — Fastest Lap")
ax3.legend()
ax3.grid(alpha=0.3)


# PLOT 4 — Throttle + Brake overlay

ax4 = axes[1, 1]
for driver, tel in telemetry_data.items():
    color = fastf1.plotting.get_driver_color(driver, session)
    ax4.plot(tel["Distance"], tel["Throttle"],
             label=f"{driver} Throttle", color=color, linewidth=1, alpha=0.8)
    ax4.fill_between(tel["Distance"], 0,
                     tel["Brake"].astype(float) * 100,
                     alpha=0.15, color=color)

ax4.set_xlabel("Distance (m)")
ax4.set_ylabel("Throttle % / Brake overlay")
ax4.set_title("Throttle & Brake Trace")
ax4.legend(fontsize=8)
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("monaco_race_analysis.png", dpi=150, bbox_inches="tight")
print("\n✓ Plot saved: monaco_race_analysis.png")


# 7. EXPORT — CSVs for further analysis

fastest_per_driver.to_csv("fastest_laps.csv", index=False)
focus_laps_clean[["Driver", "LapNumber", "Sector1Sec", "Sector2Sec",
                   "Sector3Sec", "LapSec", "Compound", "TyreLife"]].to_csv(
    "sector_times.csv", index=False
)

if telemetry_data:
    pd.concat(telemetry_data.values()).to_csv("telemetry.csv", index=False)

print("✓ CSVs exported: fastest_laps.csv, sector_times.csv, telemetry.csv")
print("\nDone.\n")
