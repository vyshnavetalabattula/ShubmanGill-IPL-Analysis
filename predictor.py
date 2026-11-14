"""Forecasting engine behind the Streamlit app (no Streamlit imports, so it is unit-testable).

Trains a Random Forest on the player's match history using only features a user can supply
before a match, and turns walk-forward residuals into prediction intervals and score probabilities.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

from features import K, build_features

warnings.filterwarnings("ignore")
DATA = Path(__file__).parent / "data" / "SubhmanGill.xlsx"
START, WARM, SEED = 40, 15, 42          # first forecast after 40 matches; intervals need 15 prior residuals

FORM = ["prev_runs", "roll3_mean", "roll5_mean", "roll5_median", "roll5_std", "roll5_fifties",
        "prev_50plus", "prev_single_digit", "career_mean", "form_vs_career"]
ENCODED = ["opp_te", "opp_n_prior", "venue_te", "venue_n_prior"]
SEQUENCE = ["season", "match_no_career", "match_no_season", "days_since_prev", "is_playoff"]
CONTEXT = ["Home_Away_Home", "Home_Away_Neutral", "Toss_WL_W", "MatchType_N", "PitchReport_Batting",
           "PitchReport_Bowling", "Captain_EionMorgan", "Captain_HardikPandya", "Captain_SubhmanGill",
           "toss_decision_field"]
APP_FEATURES = FORM + ENCODED + SEQUENCE + CONTEXT + ["bat_pos"]
CAPTAINS = ["DineshKarthik", "EionMorgan", "HardikPandya", "SubhmanGill"]   # DineshKarthik = reference level (all zeros)


def _make_model() -> object:
    return make_pipeline(SimpleImputer(strategy="median"),
                         RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5,
                                               random_state=SEED, n_jobs=-1))


@dataclass
class Forecast:
    point: float
    baseline: float
    lo80: float
    hi80: float
    lo90: float
    hi90: float
    p_single_digit: float
    p_30_plus: float
    p_50_plus: float
    features: pd.Series


class Forecaster:
    def __init__(self, data_path: Path | str = DATA):
        raw = pd.read_excel(data_path).sort_values("StartDate").reset_index(drop=True)
        raw["Opp"] = raw.Opposition.str.replace("v ", "", regex=False)
        self.raw = raw
        feats = build_features(raw.drop(columns=["Opp"]))
        feats["bat_pos"] = raw.Batting_pos.values
        missing = [c for c in APP_FEATURES if c not in feats.columns]
        assert not missing, f"feature columns missing: {missing}"
        self.X = feats[APP_FEATURES]
        self.y = feats.Bat1.values.astype(float)
        self.n = len(raw)
        # --- choices for the UI ---
        self.oppositions = raw.Opp.value_counts().index.tolist()
        self.venues = raw.Ground.value_counts().index.tolist()
        self.default_recent = [int(v) for v in self.y[-5:][::-1]]          # most recent first
        self.last_date = raw.StartDate.iloc[-1]
        self._walk_forward()
        self.model = _make_model().fit(self.X.values, self.y)

    # ------------------------------------------------------------------ validation
    def _walk_forward(self) -> None:
        idx = np.arange(START, self.n)
        est = _make_model()
        preds = np.array([clone(est).fit(self.X.values[:t], self.y[:t]).predict(self.X.values[t:t + 1])[0] for t in idx])
        base = np.array([self.y[:t].mean() for t in idx])
        act = self.y[idx]
        self.resid = act - preds                                            # actual - predicted
        ae_model, ae_base = np.abs(self.resid), np.abs(act - base)
        # expanding-window 80% interval coverage (uses only earlier residuals), as in the reliability analysis
        hits = []
        for i in range(WARM, len(idx)):
            ql, qh = np.quantile(self.resid[:i], [0.1, 0.9])
            hits.append(max(0, preds[i] + ql) <= act[i] <= preds[i] + qh)
        self.metrics = {
            "n_forecasts": len(idx), "mae_model": float(ae_model.mean()), "mae_baseline": float(ae_base.mean()),
            "rmse_model": float(np.sqrt((self.resid ** 2).mean())), "rmse_baseline": float(np.sqrt(((act - base) ** 2).mean())),
            "coverage80": float(np.mean(hits)), "n_coverage": len(hits),
            "first": self.raw.StartDate.iloc[START], "last": self.raw.StartDate.iloc[-1],
        }

    # ------------------------------------------------------------------ feature construction
    def _te(self, mask: np.ndarray) -> tuple[float, int]:
        g = self.y.mean()                                                   # global mean at the next match
        sub = self.y[mask]
        return (sub.sum() + K * g) / (len(sub) + K), int(len(sub))

    def build_row(self, *, opposition: str, venue: str | None, home_away: str, bat_pos: int, captain: str,
                  pitch: str, match_type: str, toss_won: bool, toss_decision_code: int, playoff: bool,
                  season: int, match_no_season: int, recent_scores: list[float],
                  career_mean: float | None = None, days_since_prev: float = 5) -> pd.Series:
        s = np.asarray(recent_scores, dtype=float)
        if s.shape != (5,) or (s < 0).any():
            raise ValueError("Enter exactly 5 non-negative recent scores (most recent first).")
        if home_away not in ("Home", "Away", "Neutral") or pitch not in ("Batting", "Balanced", "Bowling"):
            raise ValueError("Unknown home/away or pitch value.")
        if captain not in CAPTAINS or match_type not in ("N", "D/N"):
            raise ValueError("Unknown captain or match type.")
        career = float(self.y.mean() if career_mean is None else career_mean)
        opp_te, opp_n = self._te((self.raw.Opp == opposition).values)
        ven_te, ven_n = self._te((self.raw.Ground == venue).values) if venue else (self.y.mean(), 0)
        row = {
            "prev_runs": s[0], "roll3_mean": s[:3].mean(), "roll5_mean": s.mean(), "roll5_median": np.median(s),
            "roll5_std": s.std(ddof=1), "roll5_fifties": float((s >= 50).sum()), "prev_50plus": float(s[0] >= 50),
            "prev_single_digit": float(s[0] <= 9), "career_mean": career, "form_vs_career": s.mean() - career,
            "opp_te": opp_te, "opp_n_prior": opp_n, "venue_te": ven_te, "venue_n_prior": ven_n,
            "season": season, "match_no_career": self.n + 1, "match_no_season": match_no_season,
            "days_since_prev": days_since_prev, "is_playoff": int(playoff),
            "Home_Away_Home": int(home_away == "Home"), "Home_Away_Neutral": int(home_away == "Neutral"),
            "Toss_WL_W": int(toss_won), "MatchType_N": int(match_type == "N"),
            "PitchReport_Batting": int(pitch == "Batting"), "PitchReport_Bowling": int(pitch == "Bowling"),
            "Captain_EionMorgan": int(captain == "EionMorgan"), "Captain_HardikPandya": int(captain == "HardikPandya"),
            "Captain_SubhmanGill": int(captain == "SubhmanGill"), "toss_decision_field": int(toss_decision_code == 2),
            "bat_pos": int(bat_pos),
        }
        return pd.Series(row)[APP_FEATURES]

    # ------------------------------------------------------------------ forecast
    def predict(self, **inputs) -> Forecast:
        row = self.build_row(**inputs)
        point = float(self.model.predict(row.values.reshape(1, -1))[0])
        q05, q10, q90, q95 = np.quantile(self.resid, [0.05, 0.10, 0.90, 0.95])
        samples = np.clip(point + self.resid, 0, None)                      # forecast + historical errors
        return Forecast(
            point=point, baseline=float(self.y.mean()),
            lo80=max(0.0, point + q10), hi80=point + q90, lo90=max(0.0, point + q05), hi90=point + q95,
            p_single_digit=float((samples <= 9).mean()), p_30_plus=float((samples >= 30).mean()),
            p_50_plus=float((samples >= 50).mean()), features=row)

    # ------------------------------------------------------------------ descriptive context
    def history(self, column: str, value: str) -> pd.DataFrame:
        col = "Opp" if column == "Opposition" else "Ground"
        h = self.raw.loc[self.raw[col] == value, ["StartDate", "Opp", "Ground", "Bat1", "OutType"]].copy()
        h["StartDate"] = h.StartDate.dt.date
        return h.rename(columns={"Opp": "Opposition", "Bat1": "Runs", "OutType": "How out"}).iloc[::-1].reset_index(drop=True)
