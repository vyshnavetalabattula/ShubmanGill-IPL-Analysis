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

- **Coverage:** 86 innings
- **Date range:** 2018–2025
- **Target:** `Bat1` (runs scored in the innings)

The dataset is player-specific and relatively small, so results should be interpreted as an exploratory case study.

Before redistributing the raw spreadsheet, check the source website's terms and applicable data rights. If permission to redistribute cannot be confirmed, do not commit the spreadsheet to this public repository. Instead, place a permitted local copy at `data/SubhmanGill.xlsx` and follow `data/README.md`.

## Methodology

### Data preparation and EDA

The preparation workflow checks date ordering, duplicate records, missing values, categorical values, and batting-score distributions.

### Temporal feature engineering

Features include prior-innings scores, rolling batting averages, career-to-date summaries, match sequence information, and smoothed historical opposition/venue statistics.

The intended rule is that a match's features must not use that same match's batting outcome. The feature-engineering workflow includes a perturbation-based leakage check.

### Model comparison

Regression models are evaluated with chronological walk-forward validation. At each forecast step, models are fitted using earlier matches and predict the next held-out match.

Historical mean, historical median, recent-form mean, and previous score are included as baselines. Metrics include MAE, RMSE, and median absolute error.

### Reliability analysis

The project examines forecast residuals, large errors, and empirical prediction-interval coverage. Intervals are estimated from earlier forecast errors. Observed coverage may differ from nominal 80% or 90% levels, especially with a small evaluation sample.

### Interactive application

The Streamlit application accepts match context and recent-form inputs, then displays a model forecast, historical-average baseline, empirical ranges, score-threshold probabilities, and historical records.

## Work Completed

The following components have been implemented as part of the current CricketPulse project:

### 1. Data Collection and Preparation
- Independently collected and structured 86 Shubman Gill IPL innings records from 2018–2025.
- Prepared a chronological player-level dataset.
- Performed data-quality checks for missing values, duplicate records, date ordering, and categorical values.

### 2. Exploratory Data Analysis
- Analyzed batting-score distributions and historical performance patterns.
- Generated visualizations to understand scoring trends and batting performance.

### 3. Temporal Feature Engineering
- Engineered prior-innings score features.
- Created rolling batting averages and career-to-date statistics.
- Developed match-sequence features and historical opposition/venue statistics.
- Implemented a perturbation-based check to identify potential data leakage.

### 4. Model Development and Evaluation
- Developed and compared regression models using chronological walk-forward validation.
- Evaluated predictions using MAE, RMSE, and median absolute error.
- Compared machine-learning models against historical statistical baselines.

### 5. Forecast Reliability Analysis
- Analyzed prediction residuals and large forecasting errors.
- Generated empirical prediction intervals.
- Evaluated interval coverage and forecast reliability.

### 6. Interactive Application
- Developed an interactive Streamlit application for next-match batting-score forecasting.
- Incorporated match-context and recent-form inputs.
- Displayed forecasts, historical baselines, prediction ranges, score-threshold probabilities, and historical records.

### 7. Results and Documentation
- Generated evaluation CSV files and five analytical visualizations.
- Organized project scripts, tests, results, and documentation in GitHub.

## Ongoing Work and Planned Improvements

The next phase of CricketPulse focuses on extending the existing dataset and automating the data acquisition and model-update workflow.

### 1. Automated Scorecard Data Extraction
- Extract additional match-level attributes from individual 2026 IPL scorecards.
- Retrieve information such as toss winner, toss decision, batting position, batting partner, and partnership runs.
- Identify relevant information from scorecard tables and match details.

### 2. Data Validation and Integration
- Validate extracted attributes against their original scorecards.
- Handle missing, inconsistent, and incorrectly extracted values.
- Integrate validated records into the existing structured dataset.

### 3. Automated Feature Generation
- Extend the feature-engineering pipeline to accommodate newly extracted attributes.
- Generate updated temporal and match-context features as new records become available.

### 4. Automated Model Updating
- Develop a workflow for retraining forecasting models when new validated match records are added.
- Re-evaluate model performance using chronological validation.
- Compare updated models against historical baselines.

### 5. Application Enhancements
- Explore integrating newly collected match information into the Streamlit application.
- Improve forecast reporting and reliability analysis as the dataset grows.

**Development status:** These improvements are planned or currently under development. They should not be considered completed or validated until their implementation and testing are finished.

## Key Evaluation Finding

The walk-forward comparison found that the historical-mean baseline achieved lower MAE than the tested machine-learning models on the evaluated forecasts. This project reports that result rather than claiming machine learning improved forecast accuracy.

Prediction intervals also require cautious interpretation. They are empirical ranges derived from a limited history of forecast errors, not guaranteed confidence intervals.

## Results and Visualizations

The `Results/` directory contains generated evaluation outputs and visualizations.

### Evaluation outputs

- `evaluation_metrics.csv` – evaluation metrics for the forecasting models.
- `features_target.csv` – target-related feature dataset.
- `features_v2.csv` – engineered feature dataset.
- `interval_coverage.csv` – empirical prediction-interval coverage results.
- `model_comparison.csv` – model performance comparison.
- `predictions_with_intervals.csv` – predictions with empirical intervals.
- `residual_diagnostics.csv` – residual diagnostic summaries.
- `walkforward_predictions.csv` – chronological walk-forward predictions.
- `worst_predictions.csv` – examples of the largest prediction errors.

### Visualizations

The `Results/figures/` directory contains:

- `actual_vs_predicted_scatter.png` – actual versus predicted batting scores.
- `actual_vs_predicted_timeseries.png` – actual and predicted scores across the evaluation timeline.
- `bat1_distribution.png` – distribution of batting scores.
- `model_comparison.png` – comparison of model evaluation metrics.
- `reliability_analysis.png` – forecast residual and reliability analysis.

## Repository Structure

```text
ShubmanGill-IPL-Analysis/
│
├── README.md
├── .gitignore
├── requirements.txt
├── app.py
├── features.py
├── predictor.py
│
├── data/
│   └── README.md
│
├── Scripts/
│   ├── day1_prepare.py
│   ├── day1_features.py
│   ├── day2_models.py
│   ├── day3_eval.py
│   └── day3_reliability.py
│
├── Tests/
│   ├── test_predictor.py
│   └── smoke_test_app.py
│
└── Results/
    ├── evaluation_metrics.csv
    ├── features_target.csv
    ├── features_v2.csv
    ├── interval_coverage.csv
    ├── model_comparison.csv
    ├── predictions_with_intervals.csv
    ├── residual_diagnostics.csv
    ├── walkforward_predictions.csv
    ├── worst_predictions.csv
    │
    └── figures/
        ├── actual_vs_predicted_scatter.png
        ├── actual_vs_predicted_timeseries.png
        ├── bat1_distribution.png
        ├── model_comparison.png
        └── reliability_analysis.png
```

## Setup

Python 3.10 or later is recommended.

```bash
git clone https://github.com/vyshnavetalabattula/ShubmanGill-IPL-Analysis.git
cd ShubmanGill-IPL-Analysis

python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

Place the permitted dataset copy at `data/SubhmanGill.xlsx`.

## Run the Application

```bash
streamlit run app.py
```

## Run Tests

```bash
python Tests/test_predictor.py
python Tests/smoke_test_app.py
```

The predictor test requires the dataset at `data/SubhmanGill.xlsx`. The smoke test uses a Streamlit stub to check application execution; it does not replace testing the actual interface in a browser.

## Reproduce the Analysis

Run these commands from the repository root after placing the dataset in `data/`:

```bash
python Scripts/day1_prepare.py data/SubhmanGill.xlsx Results
python Scripts/day1_features.py data/SubhmanGill.xlsx Results
python Scripts/day2_models.py Results/features_v2.csv Results
python Scripts/day3_eval.py Results/walkforward_predictions.csv Results
python Scripts/day3_reliability.py data/SubhmanGill.xlsx Results/walkforward_predictions.csv Results
```

The scripts generate intermediate evaluation outputs and visualizations in the output directory. The committed figures are organized under `Results/figures/`.

## Limitations

- The completed dataset contains only 86 innings for one player.
- The historical-mean baseline outperformed the tested machine-learning models on MAE in the reported walk-forward comparison.
- Performance may vary across seasons, venues, opponents, batting positions, and match conditions.
- Some application inputs, such as expected batting position and future match context, require user assumptions.
- Empirical prediction intervals use a small set of prior forecast errors and may be poorly calibrated.
- The 2026 automated scorecard extraction and retraining workflow is still under development.
- This is an educational portfolio case study, not a production-grade sports forecasting service.

## Data Source Acknowledgement

ESPNcricinfo was used as a reference source when compiling the player-level match dataset. This is an independent educational analysis and is not affiliated with or endorsed by ESPNcricinfo or the IPL.

## Author

**Talabattula Sree Varaha Vyshnave**

GitHub: https://github.com/vyshnavetalabattula
