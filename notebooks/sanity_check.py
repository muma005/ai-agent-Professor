# notebooks/sanity_check.py
# Manual Submission 0 — built by hand, not by Professor
# Purpose: confirm format, metric, and establish baseline score

import sys
sys.stdout.reconfigure(encoding='utf-8')

import polars as pl
import pandas as pd  # sklearn needs pandas for now — fine in baseline script
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Load data ─────────────────────────────────────────────────────
train = pl.read_csv("data/spaceship_titanic/train.csv")
test  = pl.read_csv("data/spaceship_titanic/test.csv")
sample = pl.read_csv("data/spaceship_titanic/sample_submission.csv")

print("Train shape:", train.shape)
print("Test shape: ", test.shape)
print("\nColumns:", train.columns)
print("\nTarget distribution:")
print(train["Transported"].value_counts())
print("\nSample submission format:")
print(sample.head(3))

# ── Minimal preprocessing ─────────────────────────────────────────
# Convert to pandas for sklearn
train_pd = train.to_pandas()
test_pd  = test.to_pandas()

# Drop high-cardinality and complex columns for this baseline
drop_cols = ["PassengerId", "Name", "Cabin"]
train_pd = train_pd.drop(columns=drop_cols, errors="ignore")
test_pd  = test_pd.drop(columns=drop_cols, errors="ignore")

# Encode target
y = train_pd["Transported"].astype(int)
train_pd = train_pd.drop(columns=["Transported"])

# Force all non-numeric columns to string (handles mixed bool/NaN/str)
for col in train_pd.select_dtypes(include=["object", "bool"]).columns:
    train_pd[col] = train_pd[col].astype(str).replace("nan", "missing").replace("None", "missing")
    test_pd[col]  = test_pd[col].astype(str).replace("nan", "missing").replace("None", "missing")

# Label encode categoricals
categorical_cols = train_pd.select_dtypes(include="object").columns.tolist()
le = LabelEncoder()
for col in categorical_cols:
    # Fit on combined to avoid unseen labels
    combined = pd.concat([train_pd[col], test_pd[col]])
    le.fit(combined)
    train_pd[col] = le.transform(train_pd[col])
    test_pd[col]  = le.transform(test_pd[col])

# Fill numeric nulls
train_pd = train_pd.fillna(train_pd.median(numeric_only=True))
test_pd  = test_pd.fillna(test_pd.median(numeric_only=True))

print("\nFeatures used:", list(train_pd.columns))
print("Training shape:", train_pd.shape)

# ── Train single model — default params, no tuning ────────────────
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)
model.fit(train_pd, y)

# ── Quick local CV estimate ────────────────────────────────────────
cv_scores = cross_val_score(
    LGBMClassifier(n_estimators=500, learning_rate=0.05,
                   random_state=42, verbose=-1),
    train_pd, y,
    cv=5,
    scoring="accuracy"
)
print(f"\nLocal CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ── Generate submission ────────────────────────────────────────────
preds = model.predict(test_pd)

submission = pd.DataFrame({
    "PassengerId": test["PassengerId"].to_list(),
    "Transported": preds.astype(bool)
})

submission.to_csv("outputs/submission_0_manual.csv", index=False)
print("\nSubmission saved: outputs/submission_0_manual.csv")
print("\nSubmission format check:")
print(submission.head(5))
print(f"\nShape: {submission.shape}")
print(f"Expected: ({len(test)}, 2)")

# ── Format validation ──────────────────────────────────────────────
assert set(submission.columns) == {"PassengerId", "Transported"}, \
    "Wrong columns in submission"
assert len(submission) == len(test), \
    f"Wrong row count: {len(submission)} vs {len(test)}"
assert submission["Transported"].dtype == bool or \
    submission["Transported"].isin([True, False]).all(), \
    "Transported must be boolean"

print("\n[OK] Submission format valid")
print("[OK] Ready to submit to Kaggle")
