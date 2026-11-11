"""
app.py - a small web app around the risk model.

Run with:
    streamlit run app.py

Upload any CSV (or use the synthetic sample), choose the label/feature
columns, train, and explore the results.
"""

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve

from risk_core import (make_synthetic_data, metrics_at_threshold,
                       predict_risk, train_and_evaluate)

st.set_page_config(page_title="Athlete Risk Model", layout="wide")
st.title("Athlete High-Risk Day Predictor")
st.caption(
    "Logistic regression with feature scaling, class weighting, and an "
    "athlete-level train/test split. Bring your own CSV or use the synthetic sample."
)
st.caption("Demo of a modeling pipeline on synthetic data. Results are not validated on real "
           "athletes. See the **About & limitations** tab.")


@st.cache_data
def get_synthetic(n_athletes: int, n_days: int) -> pd.DataFrame:
    return make_synthetic_data(n_athletes=n_athletes, n_days=n_days)


def guess(columns, keywords, default=None):
    """Pick the first column whose name contains one of the keywords."""
    for kw in keywords:
        for c in columns:
            if kw in str(c).lower():
                return c
    return default


# ------------------------------------------------------------------
# Sidebar: data source
# ------------------------------------------------------------------
st.sidebar.header("1. Data")
source = st.sidebar.radio("Data source", ["Synthetic sample", "Upload CSV"])

if source == "Synthetic sample":
    n_athletes = st.sidebar.slider("Athletes", 10, 100, 40, step=5)
    n_days = st.sidebar.slider("Days per athlete", 30, 120, 60, step=10)
    df = get_synthetic(n_athletes, n_days)
    st.sidebar.download_button(
        "Download this sample as CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="sample_athlete_data.csv",
        mime="text/csv",
    )
else:
    st.sidebar.warning(
        "Privacy: don't upload personal, identifiable, or confidential athlete data "
        "to a public demo. Use synthetic or fully de-identified data only."
    )
    uploaded = st.sidebar.file_uploader("Upload a CSV file", type="csv")
    if uploaded is None:
        st.info("Upload a CSV in the sidebar to begin, or switch to the synthetic sample. "
                "Your file needs one row per athlete-day, numeric feature columns, "
                "and a column with two values (e.g. 0/1) marking high-risk days.")
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Couldn't read that file as a CSV: {e}")
        st.stop()

if df.empty or df.shape[1] < 2:
    st.error("The dataset needs at least 2 columns and 1 row.")
    st.stop()

# ------------------------------------------------------------------
# Sidebar: column choices
# ------------------------------------------------------------------
st.sidebar.header("2. Columns")
all_cols = list(df.columns)
numeric_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]

label_default = guess(all_cols, ["risk", "injur", "label", "target"], all_cols[-1])
label_col = st.sidebar.selectbox("Label (what to predict)", all_cols,
                                 index=all_cols.index(label_default))

group_options = ["(none - random split)"] + [c for c in all_cols if c != label_col]
group_default = guess([c for c in all_cols if c != label_col],
                      ["athlete", "player", "subject", "person"])
group_choice = st.sidebar.selectbox(
    "Athlete ID column (for athlete-level split)", group_options,
    index=group_options.index(group_default) if group_default in group_options else 0)
group_col = None if group_choice.startswith("(none") else group_choice

feature_choices = [c for c in numeric_cols if c not in (label_col, group_col)]
default_feats = [c for c in feature_choices
                 if not any(k in str(c).lower() for k in ["day", "date", "index", "id"])]
feature_cols = st.sidebar.multiselect("Features (numeric)", feature_choices,
                                      default=default_feats)

# If the label isn't already 0/1, let the user say which value means "high risk"
label_values = list(df[label_col].dropna().unique())
positive_value = None
if len(label_values) == 2 and set(label_values) != {0, 1}:
    positive_value = st.sidebar.selectbox("Which value means HIGH RISK?",
                                          sorted(label_values, key=str))

st.sidebar.header("3. Train")
test_size = st.sidebar.slider("Test set share", 0.1, 0.5, 0.25, step=0.05)
train_clicked = st.sidebar.button("Train model", type="primary")

if train_clicked:
    try:
        st.session_state["result"] = train_and_evaluate(
            df, feature_cols, label_col, group_col=group_col,
            positive_value=positive_value, test_size=test_size)
        st.session_state.pop("error", None)
    except ValueError as e:
        st.session_state.pop("result", None)
        st.session_state["error"] = str(e)

# ------------------------------------------------------------------
# Main area
# ------------------------------------------------------------------
tab_data, tab_results, tab_predict, tab_about = st.tabs(
    ["Data", "Results", "What-if prediction", "About & limitations"])

with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing cells", int(df.isna().sum().sum()))
    st.dataframe(df.head(50))
    st.write("Label distribution:")
    st.dataframe(df[label_col].value_counts(dropna=False).rename("count"))

with tab_results:
    if "error" in st.session_state:
        st.error(st.session_state["error"])
    elif "result" not in st.session_state:
        st.info("Choose your columns in the sidebar and click **Train model**.")
    else:
        r = st.session_state["result"]

        if "random" in r["split_type"].lower():
            st.warning(
                "No athlete ID column selected: rows from the same athlete can land in both "
                "train and test, which usually makes results look better than they are."
            )
        st.caption(f"Split: {r['split_type']}  |  train rows: {r['n_train']:,}  |  "
                   f"test rows: {r['n_test']:,}  |  high-risk rate: {r['positive_rate']:.0%}  |  "
                   f"rows dropped for missing values: {r['n_dropped']}")

        threshold = st.slider(
            "Decision threshold (flag a day as high-risk when predicted probability is at least this)",
            0.05, 0.95, 0.50, step=0.05)
        m = metrics_at_threshold(r["y_test"], r["probs"], threshold)

        a, b, c, d, e = st.columns(5)
        a.metric("AUC", f"{r['auc']:.3f}")
        b.metric("Recall", f"{m['recall']:.3f}")
        c.metric("Precision", f"{m['precision']:.3f}")
        d.metric("Accuracy", f"{m['accuracy']:.1%}")
        e.metric("Baseline accuracy", f"{r['baseline_acc']:.1%}")

        if r["cv_auc_mean"] is not None:
            st.write(f"Cross-validated AUC: **{r['cv_auc_mean']:.3f} +/- {r['cv_auc_std']:.3f}**")
        else:
            st.write("Cross-validated AUC: not available for this dataset (too few examples per class).")

        left, right = st.columns(2)
        with left:
            st.subheader("Confusion matrix")
            cm = pd.DataFrame(
                [[m["tn"], m["fp"]], [m["fn"], m["tp"]]],
                index=["Actually normal", "Actually high-risk"],
                columns=["Predicted normal", "Predicted high-risk"])
            st.dataframe(cm)
            st.caption("Lower the threshold: more risky days caught (recall up) but more false alarms "
                       "(precision down). Raise it for the opposite.")

            st.subheader("ROC curve")
            fpr, tpr, _ = roc_curve(r["y_test"], r["probs"])
            roc_df = pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr})
            st.line_chart(roc_df, x="False positive rate", y="True positive rate")
        with right:
            st.subheader("Which features matter?")
            coef_df = pd.DataFrame({"coefficient": r["coefs"]}).sort_values(
                "coefficient", key=lambda s: s.abs(), ascending=False)
            st.bar_chart(coef_df)
            st.dataframe(coef_df.style.format("{:+.2f}"))
            st.caption("Features are standardized, so coefficient sizes are comparable. "
                       "Positive = raises predicted risk, negative = lowers it. "
                       "These show association in this dataset, not cause and effect.")

with tab_predict:
    if "result" not in st.session_state:
        st.info("Train a model first.")
    else:
        r = st.session_state["result"]
        st.write("Enter one day's values to get a predicted risk probability.")
        values = {}
        cols = st.columns(min(len(r["feature_cols"]), 4))
        for i, f in enumerate(r["feature_cols"]):
            lo, hi = r["feature_ranges"][f]
            values[f] = cols[i % len(cols)].number_input(
                f, value=float(r["feature_medians"][f]),
                help=f"Range in your data: {lo:.2f} to {hi:.2f}")
        p = predict_risk(r, values)
        st.metric("Predicted probability of a high-risk day", f"{p:.0%}")
        st.progress(min(max(p, 0.0), 1.0))
        out_of_range = [f for f in values
                        if not (r["feature_ranges"][f][0] <= values[f] <= r["feature_ranges"][f][1])]
        if out_of_range:
            st.warning("These values are outside the range the model was trained on, so the "
                       "prediction is an extrapolation: " + ", ".join(out_of_range))

with tab_about:
    st.header("About this app")
    st.markdown("""
This app trains a **logistic regression** model to flag **high-risk training days** for athletes
from numeric daily measurements (for example training load, sleep, and soreness), then shows how
well it works on athletes it has never seen.

**Source code:** https://github.com/rbn2003/Risk-ModelProject
""")

    st.subheader("How it works")
    st.markdown("""
1. **Data:** one row per athlete-day, numeric features, and a two-valued label marking high-risk days.
2. **Split:** if an athlete ID column is chosen, whole athletes are held out for testing, so the model
   is never evaluated on athletes it saw in training. Without an ID column a random split is used
   and results may look better than they are.
3. **Scaling:** features are standardized (using training data only) so they share one scale.
4. **Model:** logistic regression with class weighting, because high-risk days are usually the minority.
5. **Evaluation:** AUC, precision, recall, accuracy, a confusion matrix, an ROC curve, a
   majority-class baseline, and 5-fold cross-validation.
""")

    st.subheader("What the numbers mean")
    st.markdown("""
- **AUC:** how well the model ranks high-risk days above normal days. 0.5 is chance, 1.0 is perfect.
- **Recall:** of the truly high-risk days, the share the model flagged.
- **Precision:** of the days the model flagged, the share that were truly high-risk.
- **Threshold:** the probability cutoff for flagging a day. Lowering it raises recall and lowers
  precision; raising it does the opposite.
- **Baseline accuracy:** the accuracy of always predicting the most common class. Accuracy alone
  can be misleading when high-risk days are rare, so compare AUC, recall, and precision too.
- **Coefficients:** features are standardized, so sizes are comparable. Positive raises predicted
  risk, negative lowers it. They describe association in this dataset, not cause and effect.
""")

    st.subheader("Limitations")
    st.markdown("""
- **The built-in sample is synthetic.** Its labels come from a formula written by the author, shaped
  like the one logistic regression assumes, so it favors this model. The metrics show the pipeline
  works, not how it would perform on real athletes.
- **Each day is treated independently.** Apart from one engineered feature (today's load relative to the
  previous 7 days), the model does not use day-to-day sequences.
- **Not a medical or coaching tool.** It has not been validated on real injury data and should not
  inform real training decisions.
- **Splits are not time-ordered.** A real deployment would train on earlier data and test on later data.
- **Uploaded data:** please upload only synthetic or fully de-identified data to a public demo.
""")
