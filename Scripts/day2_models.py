"""Hours 10-12: chronological (walk-forward) model comparison for next-match Bat1.

At each test match t (t >= START) every model is refit on matches 0..t-1 only and predicts match t.
Imputation/scaling live inside the pipeline, so they are fitted on training rows only.
"""
import sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
CSV, OUT = sys.argv[1], sys.argv[2]
START = 40                      # first 40 matches = minimum history before the first forecast
SEED = 42

df = pd.read_csv(CSV, parse_dates=["StartDate"]).sort_values("StartDate").reset_index(drop=True)
y = df.Bat1.values
ALL = [c for c in df.columns if c not in ("Bat1", "StartDate")]
COMPACT = ["roll5_mean", "prev_runs", "career_mean", "venue_te", "opp_te", "match_no_career"]  # LR can't take 37 cols on ~40 rows

try:
    from xgboost import XGBRegressor
    boost_name = "XGBoost"
    boost = XGBRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.7, min_child_weight=3, random_state=SEED, verbosity=0)
except ImportError:
    boost_name = "GradientBoosting (XGBoost substitute)"
    boost = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8,
                                      min_samples_leaf=3, random_state=SEED)

models = {   # name: (estimator, feature set)
    "Linear Regression (6 feats)": (make_pipeline(SimpleImputer(strategy="median"), LinearRegression()), COMPACT),
    "Ridge (all feats, extra)":    (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=30)), ALL),
    "Random Forest":               (make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                                             random_state=SEED, n_jobs=-1)), ALL),
    boost_name:                    (make_pipeline(SimpleImputer(strategy="median"), boost), ALL),
}

idx = np.arange(START, len(df))
pred = pd.DataFrame({"StartDate": df.StartDate[idx], "actual": y[idx]}, index=idx)
# --- baselines (use only history before t) ---
pred["Historical mean (baseline)"] = [y[:t].mean() for t in idx]
pred["Historical median"] = [np.median(y[:t]) for t in idx]
pred["Last-5 mean (naive form)"] = [y[t - 5:t].mean() for t in idx]
pred["Previous score (naive)"] = [y[t - 1] for t in idx]
# --- ML models, refit every step ---
for name, (est, feats) in models.items():
    X = df[feats].values
    pred[name] = [clone(est).fit(X[:t], y[:t]).predict(X[t:t + 1])[0] for t in idx]

# --- metrics ---
base = "Historical mean (baseline)"
rows = []
rng = np.random.default_rng(0)
for c in pred.columns.drop(["StartDate", "actual"]):
    e = pred[c] - pred.actual
    ae, base_ae = e.abs().values, (pred[base] - pred.actual).abs().values
    d = base_ae - ae                                  # >0 means better than baseline
    boot = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    rows.append({"model": c, "MAE": ae.mean(), "RMSE": np.sqrt((e ** 2).mean()),
                 "MedAE": np.median(ae),
                 "R2_vs_baseline": 1 - (e ** 2).sum() / ((pred[base] - pred.actual) ** 2).sum(),
                 "MAE_gain_vs_baseline": d.mean(), "gain_CI_lo": np.percentile(boot, 2.5), "gain_CI_hi": np.percentile(boot, 97.5),
                 "MAE_first_half": ae[: len(ae) // 2].mean(), "MAE_second_half": ae[len(ae) // 2:].mean()})
res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
pd.set_option("display.width", 220)
print(f"walk-forward: {len(idx)} forecasts ({df.StartDate[idx[0]].date()} -> {df.StartDate[idx[-1]].date()}), train grows from {START} to {len(df)-1}")
print(res.round(2).to_string(index=False))
pred.round(2).to_csv(f"{OUT}/walkforward_predictions.csv", index=False)
res.round(3).to_csv(f"{OUT}/model_comparison.csv", index=False)

# --- chart ---
B, G, INK, MUT, O = "#2F6DB5", "#B8BEC6", "#1F2933", "#6B7280", "#C0562F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": G,
                     "axes.labelcolor": MUT, "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold",
                     "axes.titlecolor": INK, "axes.titlelocation": "left"})
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
a = ax[0]; r = res.iloc[::-1]
cols = [O if m == base else (B if m in models else G) for m in r.model]
a.barh(r.model, r.MAE, color=cols, height=.6)
for i, (m, v, rm) in enumerate(zip(r.model, r.MAE, r.RMSE)): a.text(v + .3, i, f"MAE {v:.1f} | RMSE {rm:.1f}", va="center", fontsize=9, color=INK)
a.set(title="Walk-forward error (lower is better; orange = baseline)", xlabel="mean absolute error (runs)", xlim=(0, res.MAE.max() * 1.55))
a = ax[1]
a.plot(pred.index - START + 1, pred.actual, color=G, marker="o", ms=4, lw=1, label="actual")
a.plot(pred.index - START + 1, pred[base], color=O, lw=1.8, label="historical mean")
best_ml = res[res.model.isin(models)].iloc[0].model
a.plot(pred.index - START + 1, pred[best_ml], color=B, lw=1.8, label=best_ml)
a.set(title="One-step-ahead forecasts vs actual", xlabel="forecast number (chronological)", ylabel="runs"); a.legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/model_comparison.png", dpi=130)
