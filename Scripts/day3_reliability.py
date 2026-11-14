"""Hours 4-5: reliability analysis of walk-forward forecasts."""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

XLSX, PRED, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
raw = pd.read_excel(XLSX).sort_values("StartDate")
p = pd.read_csv(PRED, parse_dates=["StartDate"])
p = p.merge(raw[["StartDate", "Opposition", "Ground", "Home/Away", "PitchReport", "Captain", "Batting_pos", "OutType", "BallsFaced"]],
            on="StartDate", how="left")
MODELS = ["Historical mean (baseline)", "Ridge (all feats, extra)", "Random Forest"]
FOCUS = "Random Forest"
y = p.actual

# ---------------- 1. residual diagnostics (residual = actual - predicted) ----------------
print("=== Residual diagnostics ===")
diag = []
for m in MODELS:
    r = y - p[m]
    sw = stats.shapiro(r)
    lag1 = r.autocorr(1)
    het = stats.spearmanr(p[m], r.abs())[0]
    # does error grow with the size of the actual score? (it must: scores are bounded below by 0)
    diag.append({"model": m, "mean_resid": r.mean(), "std": r.std(), "skew": stats.skew(r), "shapiro_p": sw.pvalue,
                 "lag1_autocorr": lag1, "corr(|resid|,pred)": het, "p5": r.quantile(.05), "p50": r.median(), "p95": r.quantile(.95)})
diag = pd.DataFrame(diag).round(2); print(diag.to_string(index=False)); diag.to_csv(f"{OUT}/residual_diagnostics.csv", index=False)

# ---------------- 2. which matches are inaccurate ----------------
ml = ["Ridge (all feats, extra)", "Random Forest"]
p["resid_RF"] = y - p[FOCUS]
p["abs_err_RF"] = p.resid_RF.abs()
p["mean_abs_err_all"] = np.mean([(y - p[m]).abs() for m in MODELS], axis=0)   # consensus miss across 3 models
worst = p.sort_values("mean_abs_err_all", ascending=False).head(10)
cols = ["StartDate", "Opposition", "Ground", "Home/Away", "PitchReport", "Batting_pos", "OutType", "actual", FOCUS, "resid_RF", "mean_abs_err_all"]
print("\n=== 10 matches missed most (avg abs error over baseline/Ridge/RF) ===")
print(worst[cols].round(1).to_string(index=False)); worst[cols].round(1).to_csv(f"{OUT}/worst_predictions.csv", index=False)
big = p.abs_err_RF > 40
print(f"\nRF errors > 40 runs: {big.sum()} of {len(p)} ({big.mean():.0%}); of these, actual >= 60: {(big & (y >= 60)).sum()}, actual <= 15: {(big & (y <= 15)).sum()}")
print("\nRF mean residual by actual-score band (positive = under-predicted):")
band = pd.cut(y, [-1, 9, 29, 59, 200], labels=["0-9", "10-29", "30-59", "60+"])
print(p.groupby(band, observed=True).resid_RF.agg(["count", "mean"]).round(1).T.to_string())
for c in ["PitchReport", "Home/Away"]:
    print(f"\nRF MAE by {c}:", p.groupby(c).abs_err_RF.agg(["count", "mean"]).round(1).T.to_dict("index")["mean"])

# ---------------- 3. walk-forward residual-based prediction intervals ----------------
# For forecast i, the interval uses ONLY residuals of forecasts 0..i-1 (never the current one):
#   [pred + q_lo(past resid), pred + q_hi(past resid)], clipped at 0 (runs can't be negative).
WARM = 15                      # first 15 forecasts only build the residual history
def intervals(pred, level):
    a = (1 - level) / 2
    r = (y - pred).values; lo, hi = [], []
    for i in range(len(pred)):
        if i < WARM: lo.append(np.nan); hi.append(np.nan); continue
        ql, qh = np.quantile(r[:i], [a, 1 - a])
        lo.append(max(0, pred.iloc[i] + ql)); hi.append(pred.iloc[i] + qh)
    return np.array(lo), np.array(hi)

print(f"\n=== Prediction intervals (calibrated on past residuals; evaluated on forecasts {WARM+1}-{len(p)}, n={len(p)-WARM}) ===")
rows = []
for m in MODELS:
    for level in (0.8, 0.9):
        lo, hi = intervals(p[m], level)
        ok = ((y >= lo) & (y <= hi))[WARM:]
        rows.append({"model": m, "nominal": level, "coverage": ok.mean(), "mean_width": (hi - lo)[WARM:].mean(),
                     "coverage_95%CI_lo": np.nan})
        # binomial 95% CI on coverage (small n)
        n = len(ok); k = ok.sum()
        rows[-1]["coverage_95%CI_lo"] = stats.binomtest(int(k), n).proportion_ci(method="wilson").low
        rows[-1]["coverage_95%CI_hi"] = stats.binomtest(int(k), n).proportion_ci(method="wilson").high
iv = pd.DataFrame(rows).round(2); print(iv.to_string(index=False)); iv.to_csv(f"{OUT}/interval_coverage.csv", index=False)
lo80, hi80 = intervals(p[FOCUS], 0.8); lo90, hi90 = intervals(p[FOCUS], 0.9)
p["RF_lo80"], p["RF_hi80"], p["RF_lo90"], p["RF_hi90"] = lo80, hi80, lo90, hi90
p.round(1).to_csv(f"{OUT}/predictions_with_intervals.csv", index=False)
cov_band = ((y >= lo80) & (y <= hi80))[WARM:].groupby(band[WARM:], observed=True).agg(["mean", "count"]).round(2)
print("\nRF 80% interval coverage by actual-score band:\n", cov_band.T.to_string())
print(f"Share of scores that a 'simple' width of {np.nanmean((hi80-lo80)):.0f} runs can't narrow: the 80% band is almost the whole 0-{int(y.max())} range")

# ---------------- 4. figure ----------------
B, G, INK, MUT, O = "#2F6DB5", "#B8BEC6", "#1F2933", "#6B7280", "#C0562F"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": G, "axes.labelcolor": MUT,
                     "xtick.color": MUT, "ytick.color": MUT, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left"})
fig, ax = plt.subplots(2, 2, figsize=(15, 10))
r = p.resid_RF
a = ax[0, 0]; a.hist(r, bins=np.arange(-80, 100, 10), color=B, edgecolor="white", linewidth=1.5)
a.axvline(0, color=INK, lw=1); a.axvline(r.mean(), color=O, ls="--", lw=1.2)
a.text(r.mean() + 2, a.get_ylim()[1] * .92, f"mean {r.mean():+.1f}", color=O)
a.set(title=f"1. Random Forest residuals (actual − predicted): skewed right, skew {stats.skew(r):.2f}", xlabel="runs", ylabel="forecasts")
a = ax[0, 1]; a.scatter(p[FOCUS], r, color=B, s=26, alpha=.85); a.axhline(0, color=INK, lw=1)
rho=stats.spearmanr(p[FOCUS], r)[0]
a.set(title=f"2. Residual vs predicted: high forecasts overshoot (Spearman ρ = {rho:.2f})", xlabel="predicted runs", ylabel="residual (runs)")
a = ax[1, 0]; a.bar(np.arange(1, len(p) + 1), r, color=[B if v >= 0 else O for v in r], width=.75)
for d_ in ["2023-05-26", "2024-05-10", "2025-04-21"]:   # biggest miss of each season cluster
    i = p.index[p.StartDate == d_][0]
    a.annotate(p.StartDate[i].strftime("%b %y"), (i + 1, r[i]), ha="center", va="bottom" if r[i] > 0 else "top", fontsize=8, color=INK)
a.axhline(0, color=INK, lw=.8); a.set(title="3. Residuals over time (blue = under-predicted; worst clusters labelled)", xlabel="forecast number", ylabel="residual (runs)")
a = ax[1, 1]; t = np.arange(1, len(p) + 1)
a.fill_between(t, lo90, hi90, color="#DCE8F5", label="90% interval"); a.fill_between(t, lo80, hi80, color="#9DBEE3", label="80% interval")
a.plot(t, p[FOCUS], color=B, lw=1.6, label="RF forecast"); a.plot(t, y, color=INK, marker="o", ms=3, lw=.7, label="actual")
a.axvline(WARM + .5, color=G, ls=":"); a.text(WARM + 1, 143, "intervals start", fontsize=8.5, color=MUT)
a.set(title="4. Walk-forward prediction intervals (past residuals only)", xlabel="forecast number", ylabel="runs", ylim=(0, 150)); a.legend(frameon=False, ncol=2, loc="upper right")
plt.tight_layout(); plt.savefig(f"{OUT}/reliability_analysis.png", dpi=130)
