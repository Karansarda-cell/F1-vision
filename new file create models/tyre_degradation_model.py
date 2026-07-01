"""
F1 Tyre Degradation Model — Multi-Track, Multi-Year
-----------------------------------------------------
Fits lap time degradation curves per compound across tracks/years.

Pipeline:
  1. Data collection  — FastF1, multiple races + years
  2. Feature engineering — tyre age, fuel correction, compound encoding,
                           track encoding, clean air flag
  3. Baseline          — Linear + Polynomial regression per compound
  4. ML model          — XGBoost regression (global + per-track)
  5. Evaluation        — RMSE, R², residual plots
  6. Visualisation     — Degradation curves, feature importance, 3D surface


"""

import os, warnings, logging, time
import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CACHE_DIR = ".fastf1-cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Tracks selected for degradation signal variety:
# Bahrain/Spain/Silverstone = high deg | Hungary = medium | Monaco = low
RACES = [
    # (year, race_name,       round, deg_category)
    (2024, "Bahrain",         1,    "high"),
    (2024, "Spain",           10,   "high"),
    (2024, "Silverstone",     12,   "high"),
    (2024, "Hungary",         13,   "medium"),
    (2024, "Monaco",          8,    "low"),
    (2025, "Bahrain",         2,    "high"),
    (2025, "Spain",           9,    "high"),
    (2025, "Silverstone",     12,   "high"),
    (2025, "Hungary",         13,   "medium"),
    (2025, "Monaco",          8,    "low"),
]

# Fuel burn rate — industry standard ~1.8 kg/lap → ~0.034s/lap pace gain
FUEL_EFFECT_PER_LAP = 0.034   # seconds gained per lap as fuel burns

# Compounds to model (exclude WET/INTER — different physics)
DRY_COMPOUNDS = ["SOFT", "MEDIUM", "HARD"]

# ─────────────────────────────────────────────────────────────
# 1. DATA COLLECTION
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("  F1 TYRE DEGRADATION MODEL")
print("  Multi-Track | Multi-Year | XGBoost + Polynomial")
print("="*60)
print(f"\nLoading {len(RACES)} race sessions...\n")

all_laps = []

for year, race, round_num, deg_cat in RACES:
    try:
        print(f"  Loading {year} {race}...", end=" ", flush=True)
        session = fastf1.get_session(year, race, 'R')
        session.load(telemetry=False, laps=True, weather=False)

        laps = session.laps.copy()
        laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
        laps["Year"]       = year
        laps["Race"]       = race
        laps["DegCategory"] = deg_cat

        # Tag stint number per driver
        laps = laps.sort_values(["Driver", "LapNumber"])
        laps["StintChange"] = (
            laps.groupby("Driver")["Compound"]
            .transform(lambda x: (x != x.shift()).cumsum())
        )
        laps["StintNumber"] = laps.groupby("Driver")["StintChange"].transform(
            lambda x: pd.factorize(x)[0] + 1
        )

        all_laps.append(laps)
        print(f"✓ {len(laps)} laps")

    except Exception as e:
        print(f"✗ Failed: {e}")
        continue

if len(all_laps) == 0:
    print("\nNo data loaded — check your FastF1 cache or internet connection.")
    exit(1)

raw = pd.concat(all_laps, ignore_index=True)
print(f"\nTotal raw laps: {len(raw)}")

# ─────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────
print("\n--- Feature Engineering ---")

df = raw[raw["Compound"].isin(DRY_COMPOUNDS)].copy()

# Filter outliers: SC laps, in/out laps, extreme times
df = df[
    df["LapTimeSec"].notna() &
    df["TyreLife"].notna() &
    (df["PitInTime"].isna()) &
    (df["PitOutTime"].isna()) &
    (df["LapTimeSec"] > 55) &   # no lap < 55s (SC anomalies)
    (df["LapTimeSec"] < 200)    # no lap > 200s (red flag / extreme SC)
]

# Per-race lap time normalisation:
# Raw lap times changes  track to track (Monaco ~75s, Silverstone ~90s)
# We model DELTA from  race median clean lap — isolates deg signal
df["RaceMedianLapTime"] = df.groupby(["Year", "Race"])["LapTimeSec"].transform("median")
df["LapTimeDelta"] = df["LapTimeSec"] - df["RaceMedianLapTime"]

# Fuel correction: earlier laps = heavier fuel = slower
# Correct by adding back the fuel effect (lap 1 heaviest)
df["FuelCorrectedDelta"] = df["LapTimeDelta"] + (df["LapNumber"] * FUEL_EFFECT_PER_LAP)

# Encode categoricals
le_compound = LabelEncoder()
le_race     = LabelEncoder()
le_deg      = LabelEncoder()

df["CompoundCode"] = le_compound.fit_transform(df["Compound"])
df["RaceCode"]     = le_race.fit_transform(df["Race"].astype(str) + df["Year"].astype(str))
df["DegCode"]      = le_deg.fit_transform(df["DegCategory"])

# Clean air proxy: lap within top-5 position vs backmarker
# FastF1 gives Position column
df["Position"] = pd.to_numeric(df["Position"], errors="coerce")
df["CleanAir"]  = (df["Position"] <= 5).astype(int)

# Final feature set
FEATURES = [
    "TyreLife",         # primary degradation driver
    "CompoundCode",     # SOFT/MEDIUM/HARD
    "StintNumber",      # 1st stint vs 2nd stint behaviour differs
    "LapNumber",        # proxy for fuel load (inverse)
    "RaceCode",         # track identity
    "DegCode",          # high/medium/low deg track category
    "CleanAir",         # traffic effect
]
TARGET = "FuelCorrectedDelta"

model_df = df[FEATURES + [TARGET, "Compound", "Race", "Year",
                           "TyreLife", "LapTimeSec", "RaceMedianLapTime"]].dropna()

print(f"  Clean laps for modelling : {len(model_df)}")
print(f"  Features                 : {FEATURES}")
print(f"  Target                   : {TARGET} (fuel-corrected lap time delta from race median)")
print(f"  Compounds                : {model_df['Compound'].unique()}")
print(f"  Tracks                   : {model_df['Race'].unique()}")

# ─────────────────────────────────────────────────────────────
# 3. BASELINE — Linear + Polynomial regression per compound
# ─────────────────────────────────────────────────────────────
print("\n--- Baseline Models ---")
print(f"{'Compound':<10} {'Linear RMSE':>14} {'Poly-2 RMSE':>14} {'Poly-3 RMSE':>14}")
print("-" * 55)

baseline_results = {}

for compound in DRY_COMPOUNDS:
    subset = model_df[model_df["Compound"] == compound]
    if len(subset) < 30:
        continue

    X_age = subset[["TyreLife"]].values
    y     = subset[TARGET].values

    results = {}
    for deg in [1, 2, 3]:
        pipe = Pipeline([
            ("poly", PolynomialFeatures(degree=deg, include_bias=False)),
            ("lr",   LinearRegression())
        ])
        cv_scores = cross_val_score(pipe, X_age, y,
                                    cv=KFold(5, shuffle=True, random_state=42),
                                    scoring="neg_root_mean_squared_error")
        results[deg] = -cv_scores.mean()

    baseline_results[compound] = results
    print(f"{compound:<10} {results[1]:>14.4f} {results[2]:>14.4f} {results[3]:>14.4f}")

# ─────────────────────────────────────────────────────────────
# 4. XGBOOST GLOBAL MODEL
# ─────────────────────────────────────────────────────────────
print("\n--- XGBoost Global Model ---")

X = model_df[FEATURES].values
y = model_df[TARGET].values

xgb = XGBRegressor(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0
)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_rmse = -cross_val_score(xgb, X, y, cv=cv,
                            scoring="neg_root_mean_squared_error")
cv_r2   = cross_val_score(xgb, X, y, cv=cv, scoring="r2")

print(f"  CV RMSE : {cv_rmse.mean():.4f} ± {cv_rmse.std():.4f} seconds")
print(f"  CV R²   : {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# Fit on full data for plotting
xgb.fit(X, y)

# Feature importance
feat_imp = pd.Series(xgb.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(f"\n  Feature Importances:")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 40)
    print(f"    {feat:<18} {bar} {imp:.4f}")

# ─────────────────────────────────────────────────────────────
# 5. PER-TRACK DEGRADATION CURVES
# ─────────────────────────────────────────────────────────────
print("\n--- Per-Track Degradation Slopes ---")
print(f"{'Track':<15} {'Year':<6} {'Compound':<10} {'Slope (s/lap)':>15} {'R²':>8}")
print("-" * 55)

slope_data = []
for (year, race, _, _) in RACES:
    for compound in DRY_COMPOUNDS:
        subset = model_df[
            (model_df["Race"] == race) &
            (model_df["Year"] == year) &
            (model_df["Compound"] == compound)
        ]
        if len(subset) < 10:
            continue
        X_s = subset[["TyreLife"]].values
        y_s = subset[TARGET].values
        lr  = LinearRegression().fit(X_s, y_s)
        r2  = r2_score(y_s, lr.predict(X_s))
        slope = lr.coef_[0]
        slope_data.append({
            "Race": race, "Year": year, "Compound": compound,
            "Slope": slope, "R2": r2
        })
        print(f"{race:<15} {year:<6} {compound:<10} {slope:>15.4f} {r2:>8.4f}")

slope_df = pd.DataFrame(slope_data)

# ─────────────────────────────────────────────────────────────
# 6. VISUALISATION
# ─────────────────────────────────────────────────────────────
compound_colors = {"SOFT": "#FF3333", "MEDIUM": "#FFD700", "HARD": "#EEEEEE"}

fig = plt.figure(figsize=(22, 18))
fig.patch.set_facecolor("#0f0f1a")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── A: Degradation curves per compound (poly-2 fit)
ax1 = fig.add_subplot(gs[0, :2])
ax1.set_facecolor("#1a1a2e")

for compound in DRY_COMPOUNDS:
    subset = model_df[model_df["Compound"] == compound]
    if len(subset) < 10:
        continue
    color = compound_colors[compound]
    # Scatter (sampled for readability)
    sample = subset.sample(min(300, len(subset)), random_state=42)
    ax1.scatter(sample["TyreLife"], sample[TARGET],
                color=color, alpha=0.15, s=8)
    # Poly-2 fit line
    X_plot = np.linspace(1, subset["TyreLife"].max(), 100).reshape(-1, 1)
    pipe = Pipeline([("poly", PolynomialFeatures(2)), ("lr", LinearRegression())])
    pipe.fit(subset[["TyreLife"]].values, subset[TARGET].values)
    y_plot = pipe.predict(X_plot)
    ax1.plot(X_plot, y_plot, color=color, linewidth=2.5, label=compound)

ax1.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
ax1.set_xlabel("Tyre Age (laps)", color="white")
ax1.set_ylabel("Fuel-Corrected Lap Delta (s)", color="white")
ax1.set_title("Tyre Degradation Curves — All Tracks (Poly-2 Fit)", color="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white")
ax1.tick_params(colors="white")
ax1.grid(alpha=0.1)
ax1.spines[:].set_color("#333")

# ── B: Feature importance
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor("#1a1a2e")
feat_imp.sort_values().plot(kind="barh", ax=ax2, color="#7B68EE", edgecolor="none")
ax2.set_title("XGBoost Feature Importance", color="white", fontsize=9)
ax2.tick_params(colors="white", labelsize=8)
ax2.spines[:].set_color("#333")
ax2.set_facecolor("#1a1a2e")

# ── C: Per-track degradation slope heatmap
ax3 = fig.add_subplot(gs[1, :2])
ax3.set_facecolor("#1a1a2e")

if len(slope_df) > 0:
    pivot = slope_df.pivot_table(
        index=["Race", "Year"], columns="Compound", values="Slope"
    )
    im = ax3.imshow(pivot.values, cmap="RdYlGn_r", aspect="auto")
    ax3.set_xticks(range(len(pivot.columns)))
    ax3.set_xticklabels(pivot.columns, color="white", fontsize=9)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels([f"{r[0]} {r[1]}" for r in pivot.index],
                         color="white", fontsize=8)
    ax3.set_title("Degradation Slope per Track/Compound (s/lap) — Red=High Deg",
                  color="white", fontsize=9)
    plt.colorbar(im, ax=ax3, shrink=0.8)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax3.text(j, i, f"{val:.3f}", ha="center", va="center",
                         fontsize=7, color="black" if abs(val) < 0.15 else "white")

# ── D: Residuals plot (XGBoost)
ax4 = fig.add_subplot(gs[1, 2])
ax4.set_facecolor("#1a1a2e")
y_pred = xgb.predict(X)
residuals = y - y_pred
ax4.scatter(y_pred, residuals, alpha=0.1, s=5, color="#00BFFF")
ax4.axhline(0, color="red", linewidth=1, linestyle="--")
ax4.set_xlabel("Predicted (s)", color="white")
ax4.set_ylabel("Residual (s)", color="white")
ax4.set_title("XGBoost Residuals\n(random scatter = good fit)", color="white", fontsize=9)
ax4.tick_params(colors="white")
ax4.spines[:].set_color("#333")
ax4.grid(alpha=0.1)

# ── E: 3D surface — Tyre Age vs Stint Number vs Predicted Degradation
ax5 = fig.add_subplot(gs[2, :], projection='3d')
ax5.set_facecolor("#0f0f1a")

# For SOFT compound, vary tyre age and stint number, fix other features at median
soft_df = model_df[model_df["Compound"] == "SOFT"]
if len(soft_df) > 10:
    age_range   = np.linspace(1, soft_df["TyreLife"].max(), 30)
    stint_range = np.linspace(1, 3, 10)
    AA, SS = np.meshgrid(age_range, stint_range)

    # Build prediction grid
    median_features = model_df[FEATURES].median()
    soft_code = le_compound.transform(["SOFT"])[0]

    grid_rows = []
    for a, s in zip(AA.ravel(), SS.ravel()):
        row = median_features.copy()
        row["TyreLife"]     = a
        row["StintNumber"]  = s
        row["CompoundCode"] = soft_code
        grid_rows.append(row)

    grid_X  = pd.DataFrame(grid_rows)[FEATURES].values
    grid_Z  = xgb.predict(grid_X).reshape(AA.shape)

    surf = ax5.plot_surface(AA, SS, grid_Z, cmap=cm.plasma,
                             alpha=0.85, linewidth=0)
    fig.colorbar(surf, ax=ax5, shrink=0.3, pad=0.1,
                 label="Predicted Lap Delta (s)")

ax5.set_xlabel("Tyre Age (laps)", color="white", labelpad=8)
ax5.set_ylabel("Stint Number", color="white", labelpad=8)
ax5.set_zlabel("Lap Time Delta (s)", color="white", labelpad=8)
ax5.set_title(
    "3D: SOFT Tyre Degradation — Tyre Age × Stint Number × Predicted Lap Delta\n"
    "Later stints (higher Y) show steeper degradation at the same tyre age",
    color="white", fontsize=9, pad=15
)
ax5.tick_params(colors="white", labelsize=7)
ax5.xaxis.pane.fill = ax5.yaxis.pane.fill = ax5.zaxis.pane.fill = False
ax5.view_init(elev=28, azim=225)

plt.suptitle(
    "F1 Tyre Degradation Model — Multi-Track, Multi-Year\n"
    "Linear Baseline → Polynomial → XGBoost",
    fontsize=13, fontweight="bold", color="white", y=1.01
)

plt.savefig("tyre_degradation_model.png", dpi=150,
            bbox_inches="tight", facecolor="#0f0f1a")
print(f"\n✓ Saved: tyre_degradation_model.png")

# ─────────────────────────────────────────────────────────────
# 7. EXPORT
# ─────────────────────────────────────────────────────────────
model_df.to_csv("tyre_model_dataset.csv", index=False)
slope_df.to_csv("degradation_slopes.csv", index=False)

print("✓ Saved: tyre_model_dataset.csv")
print("✓ Saved: degradation_slopes.csv")
print(f"\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
print(f"  Laps used          : {len(model_df)}")
print(f"  XGBoost CV RMSE    : {cv_rmse.mean():.4f}s (target: <1.0s)")
print(f"  XGBoost CV R²      : {cv_r2.mean():.4f} (target: >0.5)")
print(f"  Top feature        : {feat_imp.index[0]}")
print(f"\n  Next step → Level 1: feed these slopes into the")
print(f"  undercut/overcut decision model.\n")




# USAGE: python tyre_degradation_model.py
#REQUIRES: pip install fastf1 pandas numpy matplotlib scikit-learn xgboost