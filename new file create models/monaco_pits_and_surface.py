"""
Monaco F1 — Pit Stop Detail + Tyre Analysis + 3D Surface Plot
--------------------------------------------------------------
Covers:
  1. Full pit stop log — lap, stint, stationary time, tyre in/out
  2. Tyre condition per driver per lap (compound + age)
  3. 3D surface plot: Lap Time vs Lap Number vs Tyre Age
     (legitimate use case — interaction of both dimensions matters)

"""

import os, warnings, logging
import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)


# CONFIG

YEAR      = 2026   # swap to 2025 if 2026 not indexed yet
RACE_NAME = "Monaco"
CACHE_DIR = ".fastf1-cache"

# Drivers to deep-dive (all others still appear in pit log)
FOCUS_DRIVERS = ["ANT", "NOR", "LEC", "PIA", "SAI"]

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)
fastf1.plotting.setup_mpl(mpl_timedelta_support=True, misc_mpl_mods=False)


# LOAD

print(f"\nLoading {YEAR} {RACE_NAME} GP...")
session = fastf1.get_session(YEAR, RACE_NAME, 'R')
session.load(telemetry=False, laps=True, weather=False)

laps = session.laps.copy()
laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
print(f"  {len(laps['Driver'].unique())} drivers | {len(laps)} total laps\n")


# 1. FULL PIT STOP LOG

print("=" * 65)
print("  PIT STOP DETAIL")
print("=" * 65)

pit_laps = laps[laps["PitInTime"].notna()].copy()

# Stationary time = PitOutTime (next lap) - PitInTime
# FastF1 stores PitInTime on the IN lap and PitOutTime on the OUT lap
pit_in  = laps[laps["PitInTime"].notna()][
    ["Driver", "LapNumber", "PitInTime", "Compound", "TyreLife", "LapTimeSec"]
].copy()
pit_out = laps[laps["PitOutTime"].notna()][
    ["Driver", "LapNumber", "PitOutTime", "Compound", "TyreLife"]
].rename(columns={
    "LapNumber": "OutLap",
    "Compound":  "NewCompound",
    "TyreLife":  "NewTyreAge"
})

# Match in/out — each pit in lap matches the next pit out lap per driver
pit_log_rows = []
for driver in laps["Driver"].unique():
    d_in  = pit_in[pit_in["Driver"] == driver].sort_values("LapNumber")
    d_out = pit_out[pit_out["Driver"] == driver].sort_values("OutLap")

    for i, (_, row_in) in enumerate(d_in.iterrows()):
        # Find the corresponding out lap (first out lap after this in lap)
        matching_out = d_out[d_out["OutLap"] > row_in["LapNumber"]]
        if len(matching_out) == 0:
            continue
        row_out = matching_out.iloc[0]

        # Stationary time in seconds
        pit_in_time  = row_in["PitInTime"]
        pit_out_time = row_out["PitOutTime"]

        if pd.notna(pit_in_time) and pd.notna(pit_out_time):
            stationary = (pit_out_time - pit_in_time).total_seconds()
        else:
            stationary = np.nan

        pit_log_rows.append({
            "Driver":         driver,
            "StopNumber":     i + 1,
            "InLap":          int(row_in["LapNumber"]),
            "OutLap":         int(row_out["OutLap"]),
            "TyreOff":        row_in["Compound"],
            "TyreAge_Off":    row_in["TyreLife"],
            "TyreOn":         row_out["NewCompound"],
            "Stationary_s":   round(stationary, 2) if pd.notna(stationary) else None,
            "InLapTime_s":    round(row_in["LapTimeSec"], 3) if pd.notna(row_in["LapTimeSec"]) else None,
        })

pit_log = pd.DataFrame(pit_log_rows).sort_values(["Driver", "StopNumber"])

print(pit_log.to_string(index=False))

# Fastest pit stop
valid_pits = pit_log[pit_log["Stationary_s"].notna() & (pit_log["Stationary_s"] > 1.5)]
if len(valid_pits) > 0:
    fastest = valid_pits.loc[valid_pits["Stationary_s"].idxmin()]
    print(f"\n  ★ Fastest stationary time: {fastest['Driver']} "
          f"Stop {int(fastest['StopNumber'])} — {fastest['Stationary_s']}s "
          f"(Lap {int(fastest['InLap'])}→{int(fastest['OutLap'])})")

    avg_pit = valid_pits["Stationary_s"].mean()
    print(f"  ✦ Field average stationary time: {avg_pit:.2f}s")


# 2. TYRE CONDITION PER DRIVER PER LAP

print(f"\n{'='*65}")
print("  TYRE CONDITION — FOCUSED DRIVERS")
print(f"{'='*65}")

tyre_cols = ["Driver", "LapNumber", "Compound", "TyreLife",
             "FreshTyre", "LapTimeSec"]
tyre_df = laps[laps["Driver"].isin(FOCUS_DRIVERS)][tyre_cols].copy()
tyre_df = tyre_df.dropna(subset=["Compound"])
tyre_df["TyreLife"] = tyre_df["TyreLife"].astype(int)

print(tyre_df.to_string(index=False))


# 3. PLOTS

compound_colors = {
    "SOFT":   "#FF3333",
    "MEDIUM": "#FFD700",
    "HARD":   "#FFFFFF",
    "INTER":  "#00CC44",
    "WET":    "#0055FF",
}

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0f0f1a")

# ── PLOT A: Pit stop stationary times (bar chart per driver)
ax1 = fig.add_subplot(3, 2, 1)
ax1.set_facecolor("#1a1a2e")
if len(valid_pits) > 0:
    colors_bar = plt.cm.tab20.colors
    for i, (_, row) in enumerate(valid_pits.iterrows()):
        label = f"{row['Driver']} S{int(row['StopNumber'])}"
        ax1.bar(i, row["Stationary_s"],
                color=colors_bar[i % len(colors_bar)],
                edgecolor="none", width=0.7)
        ax1.text(i, row["Stationary_s"] + 0.05, f"{row['Stationary_s']:.1f}s",
                 ha="center", va="bottom", fontsize=7, color="white")

    ax1.axhline(avg_pit, color="yellow", linestyle="--",
                linewidth=1, label=f"Avg: {avg_pit:.1f}s")
    ax1.set_xticks(range(len(valid_pits)))
    ax1.set_xticklabels(
        [f"{r['Driver']}\nS{int(r['StopNumber'])}" for _, r in valid_pits.iterrows()],
        fontsize=7, color="white"
    )
ax1.set_title("Pit Stop Stationary Times", color="white", fontsize=10)
ax1.set_ylabel("Seconds", color="white")
ax1.tick_params(colors="white")
ax1.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
ax1.spines[:].set_color("#333")

# ── PLOT B: Tyre strategy timeline per driver
ax2 = fig.add_subplot(3, 2, 2)
ax2.set_facecolor("#1a1a2e")

focus_tyre = tyre_df.sort_values(["Driver", "LapNumber"])
drivers_sorted = focus_tyre["Driver"].unique()

for i, driver in enumerate(drivers_sorted):
    d = focus_tyre[focus_tyre["Driver"] == driver]
    for _, row in d.iterrows():
        c = compound_colors.get(row["Compound"], "#888888")
        ax2.scatter(row["LapNumber"], i, color=c, s=60, marker="s",
                    zorder=3, linewidths=0)

ax2.set_yticks(range(len(drivers_sorted)))
ax2.set_yticklabels(drivers_sorted, fontsize=9, color="white")
ax2.set_xlabel("Lap Number", color="white")
ax2.set_title("Tyre Strategy Timeline", color="white", fontsize=10)
ax2.tick_params(colors="white")
ax2.spines[:].set_color("#333")

# Legend for compounds
patches = [mpatches.Patch(color=v, label=k) for k, v in compound_colors.items()]
ax2.legend(handles=patches, fontsize=7, facecolor="#1a1a2e",
           labelcolor="white", loc="upper right")

# ── PLOT C: Lap time evolution with tyre compound colour
ax3 = fig.add_subplot(3, 2, 3)
ax3.set_facecolor("#1a1a2e")

for driver in FOCUS_DRIVERS:
    d = focus_tyre[focus_tyre["Driver"] == driver].dropna(subset=["LapTimeSec"])
    d = d[d["LapTimeSec"] < d["LapTimeSec"].quantile(0.95)]  # strip SC laps
    try:
        color = fastf1.plotting.get_driver_color(driver, session)
    except:
        color = None
    ax3.plot(d["LapNumber"], d["LapTimeSec"],
             label=driver, linewidth=1.4, alpha=0.85, color=color)

ax3.set_xlabel("Lap Number", color="white")
ax3.set_ylabel("Lap Time (s)", color="white")
ax3.set_title("Lap Time Evolution", color="white", fontsize=10)
ax3.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
ax3.tick_params(colors="white")
ax3.spines[:].set_color("#333")
ax3.grid(alpha=0.15)

# ── PLOT D: Tyre age vs lap time scatter per compound
ax4 = fig.add_subplot(3, 2, 4)
ax4.set_facecolor("#1a1a2e")

for compound, color in compound_colors.items():
    subset = tyre_df[
        (tyre_df["Compound"] == compound) &
        (tyre_df["LapTimeSec"].notna()) &
        (tyre_df["LapTimeSec"] < tyre_df["LapTimeSec"].quantile(0.95))
    ]
    if len(subset) > 0:
        ax4.scatter(subset["TyreLife"], subset["LapTimeSec"],
                    color=color, alpha=0.5, s=20, label=compound)
        # Trend line
        if len(subset) > 3:
            z = np.polyfit(subset["TyreLife"], subset["LapTimeSec"], 1)
            p = np.poly1d(z)
            x_range = np.linspace(subset["TyreLife"].min(), subset["TyreLife"].max(), 50)
            ax4.plot(x_range, p(x_range), color=color, linewidth=1.5, alpha=0.9)

ax4.set_xlabel("Tyre Age (laps)", color="white")
ax4.set_ylabel("Lap Time (s)", color="white")
ax4.set_title("Tyre Age vs Lap Time (by compound)", color="white", fontsize=10)
ax4.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")
ax4.tick_params(colors="white")
ax4.spines[:].set_color("#333")
ax4.grid(alpha=0.15)


# 3D SURFACE PLOT — Lap Time vs Lap Number vs Tyre Age
# WHY 3D HERE: both lap number AND tyre age independently affect
# pace. A 2D plot can only show one axis vs lap time at a time.
# The surface reveals the joint degradation landscape.

ax5 = fig.add_subplot(3, 2, (5, 6), projection='3d')
ax5.set_facecolor("#0f0f1a")

# Build grid: X = tyre age, Y = lap number, Z = interpolated lap time
surface_data = tyre_df[
    tyre_df["LapTimeSec"].notna() &
    (tyre_df["LapTimeSec"] > 60) &
    (tyre_df["LapTimeSec"] < tyre_df["LapTimeSec"].quantile(0.92))
].copy()

if len(surface_data) > 20:
    tyre_age_vals = np.linspace(
        surface_data["TyreLife"].min(),
        surface_data["TyreLife"].max(), 30
    )
    lap_vals = np.linspace(
        surface_data["LapNumber"].min(),
        surface_data["LapNumber"].max(), 30
    )

    X, Y = np.meshgrid(tyre_age_vals, lap_vals)

    # Fit a 2D polynomial surface: z = a + b*x + c*y + d*x^2 + e*y^2 + f*xy
    from numpy.polynomial import polynomial as P

    x = surface_data["TyreLife"].values
    y = surface_data["LapNumber"].values
    z = surface_data["LapTimeSec"].values

    # Design matrix for degree-2 surface
    A = np.column_stack([
        np.ones(len(x)), x, y, x**2, y**2, x*y
    ])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)

    Z = (coeffs[0]
         + coeffs[1]*X + coeffs[2]*Y
         + coeffs[3]*X**2 + coeffs[4]*Y**2
         + coeffs[5]*X*Y)

    surf = ax5.plot_surface(X, Y, Z, cmap=cm.plasma,
                             alpha=0.85, linewidth=0, antialiased=True)

    # Overlay actual data points
    ax5.scatter(x, y, z, color="white", s=4, alpha=0.3, zorder=5)

    fig.colorbar(surf, ax=ax5, shrink=0.4, aspect=10,
                 label="Lap Time (s)", pad=0.1)

    ax5.set_xlabel("Tyre Age (laps)", color="white", labelpad=8)
    ax5.set_ylabel("Lap Number", color="white", labelpad=8)
    ax5.set_zlabel("Lap Time (s)", color="white", labelpad=8)
    ax5.set_title(
        "3D Surface: Lap Time vs Tyre Age vs Lap Number\n"
        "(Use this when BOTH axes independently drive the Z outcome)",
        color="white", fontsize=10, pad=12
    )
    ax5.tick_params(colors="white", labelsize=7)
    ax5.xaxis.pane.fill = False
    ax5.yaxis.pane.fill = False
    ax5.zaxis.pane.fill = False
    ax5.grid(True, alpha=0.1)
    ax5.view_init(elev=25, azim=225)

plt.suptitle(
    f"Monaco {YEAR} GP — Pit Stop Detail, Tyre Analysis & 3D Surface",
    fontsize=13, fontweight="bold", color="white", y=1.01
)
plt.tight_layout()
plt.savefig("monaco_pits_tyres_3d.png", dpi=150, bbox_inches="tight",
            facecolor="#0f0f1a")
print(f"\n✓ Saved: monaco_pits_tyres_3d.png")

# ── EXPORTS
pit_log.to_csv("pit_stop_log.csv", index=False)
tyre_df.to_csv("tyre_condition_per_lap.csv", index=False)
print("✓ Saved: pit_stop_log.csv")
print("✓ Saved: tyre_condition_per_lap.csv")
print("\nDone.\n")
