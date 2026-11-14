"""Day 1 / Hours 1-2: dataset preparation for next-match Bat1 forecasting."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC, OUT = sys.argv[1], sys.argv[2]

# ---- 1. Load ----------------------------------------------------------
df = pd.read_excel(SRC)
df = df.rename(columns={"StrikeRate(BowvsBatsman)": "StrikeRate_BvB",
                        "Toss(W/L)": "Toss_WL", "MatchType(D/N)": "MatchType",
                        "Home/Away": "Home_Away"})

# ---- 2. Validate types & chronology ------------------------------------
assert pd.api.types.is_datetime64_any_dtype(df.StartDate)
assert df.StartDate.is_monotonic_increasing and not df.StartDate.duplicated().any()
assert not df.duplicated().any()
print("rows", len(df), "| dates", df.StartDate.min().date(), "->", df.StartDate.max().date())
print("missing:", df.isna().sum()[lambda s: s > 0].to_dict())
df["Opposition"] = df.Opposition.str.replace("v ", "", regex=False)
df["Toss_Decision"] = df.Toss_Decision.map({1: "Bat", 2: "Field"})   # coded categorical (1/2) -> labelled
df["Toss_Decision"] = df.Toss_Decision.fillna("Unknown")

# ---- 3. Leakage control: only information known BEFORE the match --------
# In-match outcomes (describe the innings being predicted) are dropped.
LEAKY = ["BallsFaced", "Partnership_runs", "PartnerShipAvg", "StrikeRate_BvB",
         "OutType", "Batting_partner", "Batting_pos", "OppBowlerType"]
# Rebuild lag features from Bat1 itself so they're guaranteed leak-free.
df["PrevScore"] = df.Bat1.shift(1)                                   # fixes a bad value at row 20
df["Last5Avg"] = df.Bat1.shift(1).rolling(5, min_periods=5).mean()   # NaN during warm-up (was 0.0)
df["Last3Avg"] = df.Bat1.shift(1).rolling(3, min_periods=3).mean()
df["Career_avg_to_date"] = df.Bat1.shift(1).expanding().mean()
df["Match_no"] = np.arange(1, len(df) + 1)

# ---- 4. Outliers / distributions ---------------------------------------
q1, q3 = df.Bat1.quantile([.25, .75]); hi = q3 + 1.5 * (q3 - q1)
print(f"Bat1: mean {df.Bat1.mean():.1f} median {df.Bat1.median():.1f} skew {df.Bat1.skew():.2f} "
      f"| IQR upper fence {hi:.1f} -> {(df.Bat1 > hi).sum()} outlier(s): {df.loc[df.Bat1 > hi, ['StartDate','Bat1']].values.tolist()}")
print("Bat1 <=10:", f"{(df.Bat1 <= 10).mean():.0%}", "| >=50:", f"{(df.Bat1 >= 50).mean():.0%}")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].hist(df.Bat1, bins=15); ax[0].set(title="Bat1 distribution (right-skewed)", xlabel="runs")
ax[1].boxplot(df.Bat1, vert=False); ax[1].set(title="Bat1 boxplot")
ax[2].plot(df.StartDate, df.Bat1, marker="o", lw=.8); ax[2].set(title="Bat1 over time")
plt.tight_layout(); plt.savefig(f"{OUT}/bat1_distribution.png", dpi=130)

# ---- 5. Categoricals ----------------------------------------------------
cats = ["Opposition", "Home_Away", "Ground", "Toss_WL", "Toss_Decision",
        "MatchType", "MatchStage", "Captain", "PitchReport"]
for c in ["Ground", "Captain", "MatchStage"]:          # group rare levels (<4 rows)
    vc = df[c].value_counts(); df[c] = df[c].where(df[c].map(vc) >= 4, "Other")
df["Is_playoff"] = (df.MatchStage != "League").astype(int)
cats.remove("MatchStage")

# ---- 6. Target / features ----------------------------------------------
y = df["Bat1"]
num = ["Match_no", "PrevScore", "Last3Avg", "Last5Avg", "Career_avg_to_date",
       "Dismissal_in_last5_matches", "Is_playoff"]
X = pd.get_dummies(df[num + cats], columns=cats, drop_first=True, dtype=int)
X.insert(0, "StartDate", df.StartDate)
print("\nDropped (leaky):", LEAKY)
print("X shape", X.shape, "| NaN per col:", X.isna().sum()[lambda s: s > 0].to_dict())

df_all = pd.concat([X, y], axis=1)
df_all.to_csv(f"{OUT}/features_target.csv", index=False)
print("saved", f"{OUT}/features_target.csv")
