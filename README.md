# Retail Repeat-Purchase Prediction

Report-oriented ML mini-project: predict which existing customers buy in the next 30 days, and rank them under a fixed contact-capacity budget. No API keys, sensors, paid services or company access required.

## Run on macOS / Linux

Use Python 3.11 or 3.12. Open Terminal in this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python pipeline.py run
```

The first run downloads the public UCI workbook if it is not bundled. Reading its million Excel rows can take several minutes. Run from a writable folder; internet is needed only for the initial download and dependency installation. Outputs overwrite only this project's outputs directory. No GPU required.

Windows: activate with `.venv\Scripts\activate` instead.

## Score an original-format workbook

```bash
python pipeline.py score --input data/raw/online_retail_II.xlsx --cutoff 2011-11-01 --budget 0.2 --output customer_scores.csv
```

This example reproduces a historical scoring date. New business data must match the documented UCI column names and be validated for distribution changes. Scores are model estimates, not guaranteed or formally calibrated probabilities. Only load the model.joblib shipped with this project or a trusted model you trained; joblib uses pickle.

## Experimental design

- 180-day historical feature window; 30-day future label window.
- Monthly snapshots June 2010–November 2011.
- Train: June 2010–June 2011; validation: July–August 2011; test: September–November 2011.
- Labels finish before the next split starts. No random splitting of customer snapshots.
- Logistic regression and random forest. Fixed hyperparameters; choose by validation mean precision at 20% capacity, then refit both on train + validation.
- Baselines: recency, spend, frequency, seeded random selection and exact random expected performance.
- Main metrics: monthly precision, recall and lift at 5%, 10%, 20%, 30%, 50% capacity. Ties use customer-ID order. Secondary metrics: average precision, ROC AUC and Brier score.
- Contact budget assumes equal cost per customer; this is not a monetary-profit optimisation model.

## Deliverables

`outputs/RESULTS.md`: measured findings and limitations.
`data_audit.json`: sequential cleaning counts.
`split_summary.csv`: period sizes and class balance.
`validation_metrics.csv`: evidence for model choice.
`monthly_budget_metrics.csv`, `budget_summary.csv`: business comparisons.
`classification_metrics.csv`: ML metrics.
`test_predictions.csv`: auditable held-out customer predictions.
`*.png`: report-ready charts.
`model.joblib`: fitted model bundle.
`run_metadata.json`: source hash, settings and library versions.

## Scope and limitations

The title is not a claim of algorithmic novelty. This is an application and comparative evaluation. Historical UK transactions are not current Indian market validation. Records are invoice lines, not independent customers. Eligibility excludes anonymous and long-dormant customers. Duplicate removal may remove genuinely repeated identical lines; this assumption is disclosed. Positive sales do not represent net revenue or profit. There are no campaign-treatment outcomes, so improved buyer identification does not establish additional sales caused by outreach. Low-propensity customers may be more persuadable; this model does not estimate that.

The codebase supplies actual experimental evidence for a later report. It does not claim live deployment or guaranteed academic marks.

## Data attribution

Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. DOI: https://doi.org/10.24432/C5CG6D
Source: https://archive.ics.uci.edu/dataset/502/online+retail+ii
Dataset licence: CC BY 4.0; retain attribution. Source workbook is unchanged. Cleaned data and derived outputs are described above.

## Included verification

The bundled run completed on the full source dataset. Six automated tests and a saved-model scoring smoke test passed. Exact tested direct dependency versions are in requirements-tested.txt. The ZIP contains the original source archive, so no dataset download is needed when it is extracted into data/raw. Python dependencies still require installation.
