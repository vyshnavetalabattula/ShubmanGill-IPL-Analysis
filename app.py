"""Streamlit app: next-match batting forecast. Run with:  streamlit run app.py"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from predictor import CAPTAINS, Forecaster

st.set_page_config(page_title="Next-match batting forecast", page_icon="🏏", layout="wide")
NEW_VENUE = "Other / new venue"


@st.cache_resource(show_spinner="Training the model on match history (about a minute the first time)…")
def load_forecaster() -> Forecaster:
    return Forecaster()


fc = load_forecaster()
m = fc.metrics

st.title("🏏 Next-match batting forecast")
st.caption(f"Random Forest trained on {fc.n} innings (to {fc.last_date:%d %b %Y}). "
           "Enter what you know before the match; the forecast is runs scored in the innings (`Bat1`).")

# ------------------------------------------------------------------ inputs
with st.form("inputs"):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Match")
        opposition = st.selectbox("Opposition", fc.oppositions)
        venue_choice = st.selectbox("Venue", fc.venues + [NEW_VENUE])
        home_away = st.radio("Home / Away / Neutral", ["Home", "Away", "Neutral"], horizontal=True)
        bat_pos = st.selectbox("Expected batting position", [1, 2, 3, 4, 5, 6, 7], index=1,
                               help="The model saw positions 3–7 only in 2018–19. Since 2022 he has always batted 1 or 2, "
                                    "so other positions are extrapolation.")
        captain = st.selectbox("Captain", CAPTAINS, index=CAPTAINS.index("SubhmanGill"),
                               help="Only captains present in the history are supported.")
    with c2:
        st.subheader("Match conditions")
        pitch = st.selectbox("Pitch report", ["Batting", "Balanced", "Bowling"])
        match_type = st.selectbox("Match type", ["N", "D/N"], format_func=lambda v: {"N": "N (night)", "D/N": "D/N (day/night)"}[v])
        toss_won = st.checkbox("Won the toss", value=False)
        toss_code = st.selectbox("Toss decision (dataset code)", [1, 2], index=1,
                                 help="Coded 1 or 2 as in your spreadsheet. The model treats 2 as 'field first' — please confirm "
                                      "that matches your data's meaning.")
        playoff = st.checkbox("Playoff / final", value=False)
    with c3:
        st.subheader("Recent form")
        st.caption("Runs in his last five innings, most recent first (pre-filled with the last five in the data).")
        labels = ["Last innings", "2 innings ago", "3 innings ago", "4 innings ago", "5 innings ago"]
        recent = [st.number_input(lab, min_value=0, max_value=250, value=v, step=1, key=f"form{i}")
                  for i, (lab, v) in enumerate(zip(labels, fc.default_recent))]
        st.subheader("Season")
        season = st.number_input("Season (year)", min_value=2018, max_value=2040, value=2026, step=1)
        match_no_season = st.number_input("Match number in the season", min_value=1, max_value=20, value=1, step=1)
    with st.expander("Advanced (auto-filled)"):
        a1, a2 = st.columns(2)
        career_mean = a1.number_input("Career average to date", min_value=0.0, max_value=200.0, value=round(float(fc.y.mean()), 1), step=0.5)
        days_since = a2.number_input("Days since previous match", min_value=1, max_value=400, value=5, step=1)
    submitted = st.form_submit_button("Forecast", type="primary")

# ------------------------------------------------------------------ results
if submitted:
    venue = None if venue_choice == NEW_VENUE else venue_choice
    res = fc.predict(opposition=opposition, venue=venue, home_away=home_away, bat_pos=int(bat_pos), captain=captain,
                     pitch=pitch, match_type=match_type, toss_won=toss_won, toss_decision_code=int(toss_code),
                     playoff=playoff, season=int(season), match_no_season=int(match_no_season),
                     recent_scores=[float(v) for v in recent], career_mean=float(career_mean), days_since_prev=float(days_since))

    st.divider()
    k1, k2, k3 = st.columns(3)
    k1.metric("Model forecast", f"{res.point:.0f} runs")
    k2.metric("Historical-average baseline", f"{res.baseline:.0f} runs")
    k3.metric("80% range", f"{res.lo80:.0f} – {res.hi80:.0f}")

    left, right = st.columns([3, 2])
    with left:
        fig, ax = plt.subplots(figsize=(7, 2.6))
        ax.barh(["90% range"], [res.hi90 - res.lo90], left=res.lo90, color="#DCE8F5", height=0.5)
        ax.barh(["80% range"], [res.hi80 - res.lo80], left=res.lo80, color="#9DBEE3", height=0.5)
        ax.axvline(res.point, color="#2F6DB5", lw=2.2, label=f"forecast {res.point:.0f}")
        ax.axvline(res.baseline, color="#C0562F", lw=1.6, ls="--", label=f"baseline {res.baseline:.0f}")
        ax.set_xlim(0, max(130, res.hi90 + 5)); ax.set_xlabel("runs")
        ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, loc="lower right")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        st.markdown("**Chance of…** (forecast + his past forecast errors)")
        st.progress(res.p_single_digit, text=f"single-digit score (≤ 9): {res.p_single_digit:.0%}")
        st.progress(res.p_30_plus, text=f"30 or more: {res.p_30_plus:.0%}")
        st.progress(res.p_50_plus, text=f"50 or more: {res.p_50_plus:.0%}")

    st.warning(
        f"**How much to trust this.** In walk-forward testing on {m['n_forecasts']} matches ({m['first']:%b %Y}–{m['last']:%b %Y}) "
        f"this model's average error was **{m['mae_model']:.1f} runs**, versus **{m['mae_baseline']:.1f}** for simply guessing his "
        f"historical average, so it does not beat the baseline. Ranges labelled 80% captured the actual score "
        f"{m['coverage80']:.0%} of the time (n = {m['n_coverage']}). Treat the forecast as a rough guide, not a prediction of the score.")

    t1, t2, t3 = st.tabs([f"Vs {opposition}", f"At {venue_choice}", "Model inputs"])
    with t1:
        h = fc.history("Opposition", opposition)
        st.write(f"{len(h)} innings: average {h.Runs.mean():.1f}, median {h.Runs.median():.0f}, highest {h.Runs.max()}.")
        st.dataframe(h, hide_index=True, use_container_width=True)
    with t2:
        if venue is None:
            st.info("No history for a new venue; the model uses his overall average for this input.")
        else:
            h = fc.history("Venue", venue)
            st.write(f"{len(h)} innings: average {h.Runs.mean():.1f}, median {h.Runs.median():.0f}, highest {h.Runs.max()}.")
            st.dataframe(h, hide_index=True, use_container_width=True)
    with t3:
        st.caption("Exactly what the model received (opposition/venue become smoothed averages of his past scores there).")
        st.dataframe(res.features.rename("value").round(2).to_frame(), use_container_width=True)
else:
    st.info("Fill in the match details and press **Forecast**.")

with st.expander("About this model and its limits"):
    st.markdown(
        f"- Trained on {fc.n} IPL innings; features use only information available before the match.\n"
        f"- Validation is walk-forward (each forecast uses only earlier matches). Model MAE {m['mae_model']:.1f} / RMSE "
        f"{m['rmse_model']:.1f} vs baseline MAE {m['mae_baseline']:.1f} / RMSE {m['rmse_baseline']:.1f}.\n"
        "- Ranges and probabilities come from the model's past forecast errors, not from a formal statistical model.\n"
        "- Recent form carries little signal in this data; the forecast mostly reflects venue, conditions and the season.\n"
        "- Small samples: most venues and opponents have only a handful of innings.")
