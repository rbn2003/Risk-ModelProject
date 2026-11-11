# Risk Model Project

A small machine learning model that predicts whether a day is "high risk" 
(based on training load, sleep, and soreness) using logistic regression.

Built as a learning project while preparing for an undergraduate research 
assistant role in human performance modeling using deep learning.

## What it does

Given three numbers for a day — training load, hours of sleep, and a 
soreness rating — the model predicts whether it's likely to be a high-risk 
day (e.g. overtraining/fatigue) or a normal day.

## Why logistic regression first

I'm still building up to deep learning, so I started here deliberately: 
logistic regression is simpler and easier to fully understand end to end 
— how a model learns from labeled examples, how to hold out test data, 
how to evaluate it properly — before adding the extra complexity of 
neural networks on top.

## How it works

1. **Data**: 28 days of sample training-load, sleep, and soreness values, 
   each labeled high-risk (1) or normal (0)
2. **Split**: 22 days used to train the model, 6 held out to test it on 
   days it's never seen
3. **Train**: `LogisticRegression().fit(X_train, y_train)` — the model 
   learns a weight for each factor
4. **Evaluate**: compare predictions on the held-out test days against 
   the real answers to measure accuracy

## Results

On this run: **83% accuracy** on the held-out test days. The model's 
learned weights matched intuition — higher training load and soreness 
pushed predicted risk up, more sleep pushed it down.

## Run it yourself

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 simple_riskmodel.py
```

## What I'd improve next

- Use a real dataset instead of hand-made sample data
- Compare against a basic neural network (e.g. scikit-learn's 
  `MLPClassifier`) to see whether it actually improves on this baseline
- Add cross-validation instead of a single train/test split, since 28 
  data points is small enough that one split can be a bit lucky/unlucky