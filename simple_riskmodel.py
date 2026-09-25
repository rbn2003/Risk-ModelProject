"""
Simple risk-prediction model using scikit-learn.

Goal: given a day's training load, sleep hours, and soreness rating,
predict whether it's a "high risk" day (1) or not (0).

This uses scikit-learn instead of raw PyTorch -- no manual training loop,
no backward-pass calculus. Just: give it data, call .fit(), call .predict().
This is a completely normal, real way to build a first model.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# -----------------------------------------------------------------
# The data: 28 days of made-up (but realistic-looking) tracking.
# Columns: training_load (0-10 scale), sleep_hours, soreness (1-10)
# Label: 1 = high-risk day, 0 = normal day
# -----------------------------------------------------------------
data = [
    # load, sleep, soreness, high_risk
    (3, 8.0, 2, 0),
    (4, 7.5, 3, 0),
    (8, 5.0, 7, 1),
    (2, 8.5, 2, 0),
    (9, 4.5, 8, 1),
    (5, 7.0, 4, 0),
    (7, 6.0, 6, 1),
    (3, 8.0, 3, 0),
    (6, 6.5, 5, 0),
    (9, 5.5, 7, 1),
    (2, 9.0, 1, 0),
    (8, 5.0, 8, 1),
    (4, 7.5, 3, 0),
    (7, 5.5, 6, 1),
    (3, 8.0, 2, 0),
    (9, 4.0, 9, 1),
    (5, 7.0, 4, 0),
    (6, 6.0, 5, 1),
    (2, 8.5, 2, 0),
    (8, 5.0, 7, 1),
    (4, 7.5, 3, 0),
    (7, 5.5, 6, 1),
    (3, 8.5, 2, 0),
    (9, 4.5, 8, 1),
    (5, 7.0, 4, 0),
    (6, 6.5, 5, 0),
    (8, 5.0, 8, 1),
    (2, 9.0, 1, 0),
]

# split each row into inputs (X) and the answer (y)
X = [[row[0], row[1], row[2]] for row in data]
y = [row[3] for row in data]

# hold out 6 days to test on, train on the rest
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=6, random_state=1
)

# -----------------------------------------------------------------
# Build and train the model. That's really it -- two lines.
# -----------------------------------------------------------------
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------------------------
# See how it does on days it never saw
# -----------------------------------------------------------------
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Predicted:", list(predictions))
print("Actual:   ", y_test)
print(f"Accuracy: {accuracy:.0%}")

# which factor mattered most to the model?
feature_names = ["training_load", "sleep_hours", "soreness"]
for name, weight in zip(feature_names, model.coef_[0]):
    print(f"  {name}: weight = {weight:.2f}")