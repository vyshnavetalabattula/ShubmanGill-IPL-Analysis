"""Hours 6-9: leak-free feature engineering for next-match Bat1 forecasting.

Rule: the features of row t may use data from rows < t only (plus pre-match context of row t).
"""
import sys
import numpy as np
import pandas as pd

K = 3  # smoothing strength for target encodings (pseudo-matches at the prior mean)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    d = raw.rename(columns={"StrikeRate(BowvsBatsman)": "StrikeRate_BvB", "Toss(W/L)": "Toss_WL",
                            "MatchType(D/N)": "MatchType", "Home/Away": "Home_Away"}).copy()
    d = d.sort_values("StartDate").reset_index(drop=True)
    d["Opposition"] = d.Opposition.str.replace("v ", "", regex=False)
    f = pd.DataFrame({"StartDate": d.StartDate})
    prior = d.Bat1.shift(1)                       # every history feature starts from this

    # --- previous innings / previous match performance -------------------
    f["prev_runs"] = prior
    f["prev_balls"] = d.BallsFaced.shift(1)
    f["prev_sr"] = d.Bat1.shift(1) / d.BallsFaced.shift(1) * 100          # his own strike rate last match
    f["prev_not_out"] = (d.OutType.shift(1) == "NotOut").astype(float).where(d.OutType.shift(1).notna())
    f["prev_partnership"] = d.Partnership_runs.shift(1)
    f["prev_50plus"] = (prior >= 50).astype(float).where(prior.notna())
    f["prev_single_digit"] = (prior <= 9).astype(float).where(prior.notna())

    # --- rolling form (window ends at t-1) -------------------------------
    for w in (3, 5):
        f[f"roll{w}_mean"] = prior.rolling(w, min_periods=w).mean()
        f[f"roll{w}_median"] = prior.rolling(w, min_periods=w).median()
        f[f"roll{w}_std"] = prior.rolling(w, min_periods=w).std()
    f["roll10_mean"] = prior.rolling(10, min_periods=5).mean()
    f["roll5_fifties"] = (prior >= 50).astype(float).where(prior.notna()).rolling(5, min_periods=5).sum()
    f["career_mean"] = prior.expanding().mean()
    f["form_vs_career"] = f.roll5_mean - f.career_mean

    # --- sequence / season -------------------------------------------------
    f["season"] = d.StartDate.dt.year
    f["match_no_career"] = np.arange(1, len(d) + 1)
    f["match_no_season"] = d.groupby(d.StartDate.dt.year).cumcount() + 1
    f["days_since_prev"] = d.StartDate.diff().dt.days
    f["is_playoff"] = (d.MatchStage != "League").astype(int)
    f["season_mean_so_far"] = (prior.groupby(d.StartDate.dt.year).transform(lambda s: s.expanding().mean()))
    f.loc[f.match_no_season == 1, "season_mean_so_far"] = np.nan   # prior-season innings excluded on purpose

    # --- opposition & venue encodings (expanding, smoothed target encoding) --
    gmean = prior.expanding().mean()                       # global prior mean available at t
    for col, name in [("Opposition", "opp"), ("Ground", "venue")]:
        g = d.groupby(col)
        s_prior = g.Bat1.cumsum() - d.Bat1                 # sum of earlier innings vs this opp/at this venue
        n_prior = g.cumcount()                             # count of earlier innings
        f[f"{name}_n_prior"] = n_prior
        f[f"{name}_te"] = (s_prior + K * gmean) / (n_prior + K)   # shrinks to global mean when n_prior small
        f[f"{name}_te"] = f[f"{name}_te"].where(gmean.notna())    # undefined for the very first match
    # opposition x ground-type pre-match context (known before the first ball)
    for c in ["Home_Away", "Toss_WL", "MatchType", "PitchReport", "Captain"]:
        vc = d[c].value_counts()
        col = d[c].where(d[c].map(vc) >= 4, "Other")       # rare levels grouped
        f = pd.concat([f, pd.get_dummies(col, prefix=c, drop_first=True, dtype=int)], axis=1)
    f["toss_decision_field"] = (d.Toss_Decision == 2).astype(int)

    f["Bat1"] = d.Bat1                                     # target, kept last, never used as a feature
    return f


def leakage_test(raw: pd.DataFrame, f: pd.DataFrame, n_trials: int = 20, seed: int = 0) -> None:
    """Perturb Bat1 (and in-match stats) at match t; features at t and earlier must not change."""
    rng = np.random.default_rng(seed)
    feats = [c for c in f.columns if c != "Bat1"]
    for t in rng.choice(len(raw), n_trials, replace=False):
        r2 = raw.copy()
        for c in ["Bat1", "BallsFaced", "Partnership_runs", "PartnerShipAvg", "StrikeRate(BowvsBatsman)"]:
            r2.loc[t, c] = r2.loc[t, c] + 500
        r2.loc[t, "OutType"] = "Bowled" if raw.loc[t, "OutType"] == "NotOut" else "NotOut"
        f2 = build_features(r2)
        a, b = f.loc[:t, feats], f2.loc[:t, feats]
        diff = ~((a == b) | (a.isna() & b.isna())).all(axis=1)
        assert not diff.any(), f"LEAK at t={t}: rows {list(diff[diff].index)}"
    print(f"leakage test passed ({n_trials} random perturbations): features at t never depend on match t's outcome")


if __name__ == "__main__":
    src, out = sys.argv[1], sys.argv[2]
    raw = pd.read_excel(src)
    f = build_features(raw)
    leakage_test(raw, f)
    # sanity: own dismissal count vs provided column
    own = (raw.OutType.shift(1) != "NotOut").astype(float).where(raw.OutType.shift(1).notna()).rolling(5, min_periods=5).sum()
    print("Dismissal_in_last5_matches in file vs recomputed-from-prior-5:",
          f"{(own.dropna() == raw.Dismissal_in_last5_matches[own.notna()]).mean():.0%} match")
    f.to_csv(f"{out}/features_v2.csv", index=False)
    feats = [c for c in f.columns if c not in ("Bat1", "StartDate")]
    print(f"saved {out}/features_v2.csv | {len(f)} rows, {len(feats)} features")
    print("NaNs per feature:", f[feats].isna().sum()[lambda s: s > 0].to_dict())
    corr = f[feats + ["Bat1"]].corr(method="spearman")["Bat1"].drop("Bat1").sort_values(key=abs, ascending=False)
    print("\nTop |Spearman| with Bat1:\n", corr.head(10).round(2).to_string())
