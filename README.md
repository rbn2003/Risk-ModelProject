# Athlete High-Risk Day Predictor

Live Demo: https://ariskmodel.streamlit.app/

A small, end-to-end machine learning project that flags **high-risk training days** for athletes from training load, sleep, and soreness. It includes a reusable model module and an interactive web app that works with your own CSV data.

> **Important:** the built-in data is **synthetic**. The labeling rule was written by me, so the reported metrics show that the pipeline works, not how well the model would perform on real athletes. See [Limitations](#limitations).

## What it does

- Trains a **logistic regression** classifier (scikit-learn) on any dataset with one row per athlete-day
- Standardizes features and uses **class weighting**, since high-risk days are the minority
- Splits train/test **by athlete**, so test athletes are never seen during training
- Reports **AUC, precision, recall, accuracy**, a confusion matrix, an ROC curve, and a majority-class baseline
- Uses 5-fold, athlete-level **cross-validation** to check that results are stable
- Shows **standardized coefficients** so features can be compared on one scale
- Lets you move a **decision threshold** to see the precision/recall tradeoff
- Includes a **what-if panel**: enter one day's values and get a predicted risk probability

## Run it

Requires Python 3.9 or newer.

```bash
git clone https://github.com/rbn2003/Risk-ModelProject.git
cd Risk-ModelProject

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. Pick **Synthetic sample** or **Upload CSV**, choose your columns in the sidebar, and click **Train model**.

To run the model without the app:

```bash
python risk_model_v2.py
```

## Using your own data

Your CSV needs:

| Requirement | Details |
|---|---|
| One row per athlete-day | Each row is one athlete on one day |
| Numeric feature columns | e.g. training load, sleep hours, soreness |
| A label column | Exactly two values (0/1, or text such as high/normal) marking high-risk days |
| *(Recommended)* An athlete ID column | Enables the athlete-level split; without it, a random split is used and the app warns you that results may be optimistic |

`sample_athlete_data.csv` shows the expected format.

## Project structure

```
app.py                  Streamlit web app (interface only)
risk_core.py            Model logic: data generation, training, evaluation, prediction
risk_model_v2.py        Standalone script version of the pipeline
risk_model_v1.py        First version: a tiny hand-made dataset (kept to show the progression)
sample_athlete_data.csv Example input file (synthetic)
requirements.txt        Dependencies
```

## How the synthetic data works

For each simulated athlete, daily load varies around a personal baseline. Sleep dips and soreness rises on hard days. One engineered feature, `load_vs_recent_avg`, is today's load divided by the athlete's previous 7-day average, a simplified version of the acute:chronic workload ratio from sports science. The chance of a high-risk day rises with load, poor sleep, soreness, and sudden load spikes, plus random noise so labels are not a perfect function of the inputs.

## Example results (synthetic data)

With the default sample (40 athletes x 60 days, 25% high-risk days), on held-out athletes:

| Metric | Value |
|---|---|
| AUC | 0.887 |
| Recall (threshold 0.5) | 0.823 |
| Precision (threshold 0.5) | 0.532 |
| Accuracy | 80.5% (majority-class baseline: 78.3%) |
| 5-fold athlete-level CV AUC | about 0.86 +/- 0.01 |

Accuracy barely beats the baseline because most days are normal, so AUC, recall, and precision are the more informative numbers. Precision around 0.53 means about half of the flagged days are false alarms, which is the tradeoff the threshold slider shows.

## Limitations

- **Synthetic data.** Labels come from a formula I wrote, shaped like the one logistic regression assumes, so this setting favors the model. Metrics do not transfer to real athletes.
- **Each day is treated independently.** The model does not use full day-to-day sequences, apart from the single `load_vs_recent_avg` feature.
- **Association, not causation.** Coefficients describe patterns in the data, not causes of injury.
- **Not a medical or coaching tool.** It has not been validated on real injury data and should not inform real training decisions.
- **Splits are not time-ordered.** A real deployment would train on earlier data and test on later data.

## Roadmap

- [ ] Test on real or public de-identified athlete-monitoring data
- [ ] Time-ordered validation (train on earlier weeks, test on later weeks)
- [ ] Compare against a sequence model (LSTM) on 21-day windows
- [ ] Per-athlete baselines and exponentially weighted workload features
- [ ] Probability calibration checks
- [ ] Deploy the app (Streamlit Community Cloud)

## Tech

Python, scikit-learn, pandas, NumPy, Streamlit
