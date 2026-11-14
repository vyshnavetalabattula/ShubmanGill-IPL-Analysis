# CricketPulse: IPL Batting Performance Forecasting & Reliability Analysis

CricketPulse is an end-to-end data science project exploring Shubman Gill's IPL batting history and investigating whether historical match information can support a next-match batting-score forecast.

The project covers dataset preparation, exploratory data analysis, temporal feature engineering, chronological model evaluation, forecast reliability analysis, and an interactive Streamlit application.

> This is an exploratory project built from a small, player-specific dataset. Forecasts are not intended for betting or other high-stakes use.

## Objectives
- Prepare and validate a chronological player-level IPL dataset.
- Explore batting-score distributions and historical patterns.
- Engineer features using information available before the innings being forecast.
- Compare regression models with historical baselines using walk-forward validation.
- Examine residuals and empirical prediction-interval coverage.
- Provide an interactive interface for exploring forecasts.

## Dataset
The dataset was independently compiled and structured from publicly available Shubman Gill IPL match records, with ESPNcricinfo used as a reference source.

- Coverage: 86 innings
- Date range: 2018–2025
- Target: `Bat1` (runs scored in the innings)

The dataset is player-specific and relatively small, so results should be interpreted as an exploratory case study.

Before redistributing the raw spreadsheet, check the source website's terms and applicable data rights. If permission to redistribute cannot be confirmed, do not commit the spreadsheet to this public repository. Instead, place a permitted local copy at `data/SubhmanGill.xlsx` and follow `data/README.md`.

## Methodology

### Data preparation and EDA
The preparation workflow checks date ordering, duplicate records, missing values, categorical values, and batting-score distributions.

### Temporal feature engineering
Features include prior-innings scores, rolling batting averages, career-to-date summaries, match sequence information, and smoothed historical opposition/venue statistics. The intended rule is that a match's features must not use that same match's batting outcome. The feature-engineering workflow includes a perturbation-based leakage check.

### Model comparison
Regression models are evaluated with chronological walk-forward validation. At each forecast step, models are fitted using earlier matches and predict the next held-out match. Historical mean, historical median, recent-form mean, and previous score are included as baselines. Metrics include MAE, RMSE, and median absolute error.

### Reliability analysis
The project examines forecast residuals, large errors, and empirical prediction-interval coverage. Intervals are estimated from earlier forecast errors. Observed coverage may differ from nominal 80% or 90% levels, especially with a small evaluation sample.

### Interactive application
The Streamlit application accepts match context and recent-form inputs, then displays a model forecast, historical-average baseline, empirical ranges, score-threshold probabilities, and historical records.

## Key evaluation finding
The walk-forward comparison found that the historical-mean baseline achieved lower MAE than the tested machine-learning models on the evaluated forecasts. This project reports that result rather than claiming machine learning improved forecast accuracy.

Prediction intervals also require cautious interpretation. They are empirical ranges derived from a limited history of forecast errors, not guaranteed confidence intervals.

## Repository structure
```text
.
├── app.py
├── features.py
├── predictor.py
├── data/
│   └── README.md
├── scripts/
│   ├── day1_prepare.py
│   ├── day1_features.py
│   ├── day2_models.py
│   ├── day3_eval.py
│   └── day3_reliability.py
├── tests/
│   ├── test_predictor.py
│   └── smoke_test_app.py
├── results/
└── reports/
    └── figures/
```

## Setup
Python 3.10 or later is recommended.

```bash
git clone https://github.com/vyshnavetalabattula/ShubmanGill-IPL-Analysis.git
cd ShubmanGill-IPL-Analysis

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate       # Windows

pip install -r requirements.txt
```

Place the permitted dataset copy at `data/SubhmanGill.xlsx`.

## Run the application
```bash
streamlit run app.py
```

## Run tests
```bash
python tests/test_predictor.py
python tests/smoke_test_app.py
```

The predictor test requires the dataset at `data/SubhmanGill.xlsx`. The smoke test uses a Streamlit stub to check application execution; it does not replace testing the actual interface in a browser.

## Reproduce the analysis
Run these commands from the repository root after placing the dataset in `data/`:

```bash
mkdir -p results reports/figures

python scripts/day1_prepare.py data/SubhmanGill.xlsx results
python scripts/day1_features.py data/SubhmanGill.xlsx results
python scripts/day2_models.py results/features_v2.csv results
python scripts/day3_eval.py results/walkforward_predictions.csv results
python scripts/day3_reliability.py data/SubhmanGill.xlsx results/walkforward_predictions.csv results
```

Some scripts save charts to the output directory. Move or copy final figures into `reports/figures/` to keep report assets separate from intermediate outputs.

## Limitations
- The dataset contains only 86 innings for one player.
- The historical-mean baseline outperformed the tested machine-learning models on MAE in the reported walk-forward comparison.
- Performance may vary across seasons, venues, opponents, batting positions, and match conditions.
- Some application inputs, such as expected batting position and future match context, require user assumptions.
- Empirical prediction intervals use a small set of prior forecast errors and may be poorly calibrated.
- This is an educational portfolio case study, not a production-grade sports forecasting service.

## Data source acknowledgement
ESPNcricinfo was used as a reference source when compiling the player-level match dataset. This is an independent educational analysis and is not affiliated with or endorsed by ESPNcricinfo or the IPL.

## Author
**Vyshnave Talabattula**

GitHub: https://github.com/vyshnavetalabattula
