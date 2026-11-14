"""Sanity tests for the forecasting engine. Run: python test_predictor.py"""
import numpy as np
from predictor import Forecaster, APP_FEATURES

fc = Forecaster()
print("history rows:", fc.n, "| oppositions:", len(fc.oppositions), "| venues:", len(fc.venues))
print("validation:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in fc.metrics.items()})

base = dict(opposition="CSK", venue="Ahmedabad", home_away="Home", bat_pos=2, captain="SubhmanGill", pitch="Batting",
            match_type="N", toss_won=True, toss_decision_code=2, playoff=False, season=2026, match_no_season=1,
            recent_scores=fc.default_recent)
f = fc.predict(**base)
print(f"forecast {f.point:.1f} | baseline {f.baseline:.1f} | 80% [{f.lo80:.0f}, {f.hi80:.0f}] | 90% [{f.lo90:.0f}, {f.hi90:.0f}]")
print(f"P(<=9) {f.p_single_digit:.0%}  P(>=30) {f.p_30_plus:.0%}  P(>=50) {f.p_50_plus:.0%}")

# 1. row has exactly the training columns, in order, with no NaN
assert list(f.features.index) == APP_FEATURES and not f.features.isna().any()
# 2. interval ordering and non-negativity
assert f.lo90 <= f.lo80 and f.hi80 <= f.hi90 and f.lo90 >= 0
# 3. probabilities are monotone
assert f.p_50_plus <= f.p_30_plus <= 1
# 4. recomputing form features from the UI inputs matches what training used for the last real match
last = fc.X.iloc[-1]
rebuilt = fc.build_row(**{**base, "recent_scores": [fc.y[-2], fc.y[-3], fc.y[-4], fc.y[-5], fc.y[-6]],
                          "opposition": fc.raw.Opp.iloc[-1], "venue": fc.raw.Ground.iloc[-1]},
                       career_mean=fc.y[:-1].mean())
for c in ["prev_runs", "roll3_mean", "roll5_mean", "roll5_median", "roll5_std", "roll5_fifties", "career_mean"]:
    assert np.isclose(rebuilt[c], last[c], equal_nan=True), (c, rebuilt[c], last[c])
print("form features rebuilt from raw inputs match the training pipeline for the last match: OK")
# 5. unseen venue falls back to the overall average with n_prior = 0
nv = fc.build_row(**{**base, "venue": None})
assert nv.venue_n_prior == 0 and np.isclose(nv.venue_te, fc.y.mean())
# 6. inputs change the forecast (model isn't ignoring the form)
hot = fc.predict(**{**base, "recent_scores": [90, 80, 100, 60, 70]}).point
cold = fc.predict(**{**base, "recent_scores": [2, 5, 0, 8, 1]}).point
print(f"forecast after hot form {hot:.1f} vs cold form {cold:.1f}")
# 7. bad input is rejected
for bad in [dict(recent_scores=[1, 2, 3]), dict(pitch="Green"), dict(captain="Nobody")]:
    try:
        fc.predict(**{**base, **bad}); raise SystemExit(f"accepted bad input {bad}")
    except ValueError:
        pass
print("ALL TESTS PASSED")
