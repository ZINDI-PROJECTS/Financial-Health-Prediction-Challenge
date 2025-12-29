"""
V24: STARTER-ALIGNED APPROACH

CRITICAL DISCOVERY: We've been using WRONG profit_margin formula!

Starter Notebook (CORRECT):
  profit_margin = (personal_income - business_expenses) / personal_income

Our V6 (WRONG):
  profit_margin = log1p(turnover / expenses)

This version:
1. Uses EXACT profit_margin from starter
2. Uses EXACT financial_access_score from starter
3. Keeps proven V10 ensemble approach (V2 + V9)
4. Minimal, clean, starter-aligned

Expected: If this is the missing piece, we should hit 0.893-0.900
"""
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("V24: STARTER-ALIGNED APPROACH")
print("="*80)
print("\nUsing CORRECT features from starter notebook:")
print("  1. profit_margin = (income - expenses) / income")
print("  2. financial_access_score")
print("  3. V10 ensemble (V2 + V9)")

# ============================================
# LOAD DATA
# ============================================
print("\n[1/5] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# ============================================
# FEATURE ENGINEERING (STARTER-ALIGNED)
# ============================================
print("\n[2/5] Engineering features (starter-aligned)...")

def engineer_starter_features(df):
    """Apply EXACT feature engineering from starter notebook"""
    df = df.copy()

    # Feature 1: CORRECT Profit Margin
    # Formula from starter: (income - expenses) / income
    profit_margin = []
    for idx, row in df.iterrows():
        income = row.get('personal_income', np.nan)
        expenses = row.get('business_expenses', np.nan)

        if pd.notna(income) and pd.notna(expenses) and income != 0:
            margin = (income - expenses) / income
            margin = max(-1, min(margin, 1))  # Cap between -1 and 1
        else:
            margin = np.nan

        profit_margin.append(margin)

    df['profit_margin'] = profit_margin

    # Feature 2: Financial Access Score
    # From starter: counts financial services usage
    financial_features = ['has_loan_account', 'has_internet_banking',
                          'has_debit_card', 'medical_insurance', 'funeral_insurance']

    scores = []
    for idx, row in df.iterrows():
        score = 0
        valid_features = 0

        for feature in financial_features:
            if feature in df.columns:
                value = row.get(feature, np.nan)

                if pd.notna(value):
                    valid_features += 1
                    if value in ['Yes', 'Have now', 'have now']:
                        score += 1
                    elif 'Used to have' in str(value):
                        score += 0.5

        if valid_features > 0:
            normalized_score = score / valid_features
        else:
            normalized_score = np.nan

        scores.append(normalized_score)

    df['financial_access_score'] = scores

    return df

train_df = engineer_starter_features(train_df)
test_df = engineer_starter_features(test_df)

print("  ✓ Created profit_margin (starter formula)")
print("  ✓ Created financial_access_score")

# ============================================
# PREPARE DATA
# ============================================
print("\n[3/5] Preparing data...")

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}

y = train_df['Target'].map(label_mapping).values
test_ids = test_df['ID']

X = train_df.drop(['ID', 'Target'], axis=1)
X_test = test_df.drop(['ID'], axis=1)

# Identify categorical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Handle missing values (simple approach)
for col in categorical_cols:
    X[col] = X[col].fillna('Missing').astype(str)
    X_test[col] = X_test[col].fillna('Missing').astype(str)

numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

print(f"  Features: {X.shape[1]}")
print(f"  Samples: {len(X):,}")

# ============================================
# TRAIN V10 ENSEMBLE (V2 + V9)
# ============================================
print("\n[4/5] Training V10 ensemble with corrected features...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

# -----------------------------------
# V2: No SMOTE
# -----------------------------------
print("  [V2] Training CatBoost...", end=' ')
v2_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
v2_cat.fit(X, y, cat_features=categorical_cols, verbose=False)
v2_cat_proba = v2_cat.predict_proba(X_test)
print("Done")

print("  [V2] Training LightGBM...", end=' ')
X_lgb = X.copy()
X_test_lgb = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_lgb[col] = le.transform(X[col])
    X_test_lgb[col] = le.transform(X_test[col])

v2_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
v2_lgb.fit(X_lgb, y)
v2_lgb_proba = v2_lgb.predict_proba(X_test_lgb)
print("Done")

v2_proba = (v2_cat_proba + v2_lgb_proba) / 2

# -----------------------------------
# V9: With SMOTE
# -----------------------------------
print("  [V9] Encoding for SMOTE...", end=' ')
X_smote = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_smote[col] = le.transform(X[col])
print("Done")

print("  [V9] Applying SMOTE...", end=' ')
sampling_strategy = {0: (y == 0).sum(), 1: (y == 1).sum(), 2: 1000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_smote_resampled, y_smote = smote.fit_resample(X_smote, y)
print("Done")

print("  [V9] Training CatBoost...", end=' ')
v9_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
v9_cat.fit(X_smote_resampled, y_smote, verbose=False)

X_test_smote = X_test.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_test_smote[col] = le.transform(X_test[col])

v9_cat_proba = v9_cat.predict_proba(X_test_smote)
print("Done")

print("  [V9] Training LightGBM...", end=' ')
v9_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
v9_lgb.fit(X_smote_resampled, y_smote)
v9_lgb_proba = v9_lgb.predict_proba(X_test_smote)
print("Done")

v9_proba = (v9_cat_proba + v9_lgb_proba) / 2

# -----------------------------------
# Blend V2 + V9 (50/50)
# -----------------------------------
v10_proba = (v2_proba + v9_proba) / 2
v10_pred = np.argmax(v10_proba, axis=1)
v10_predictions = [reverse_mapping[p] for p in v10_pred]

# ============================================
# ANALYZE & SAVE
# ============================================
print("\n[5/5] Analyzing results...")

pred_counts = pd.Series(v10_predictions).value_counts()
print(f"\nV24 predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(v10_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to original V10
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v10_counts = v10_sub['Target'].value_counts()

print(f"\nComparison to original V10:")
print(f"  Original V10: Low={v10_counts.get('Low', 0)}, Med={v10_counts.get('Medium', 0)}, High={v10_counts.get('High', 0)}")
print(f"  V24 (starter): Low={pred_counts.get('Low', 0)}, Med={pred_counts.get('Medium', 0)}, High={pred_counts.get('High', 0)}")

diff_low = pred_counts.get('Low', 0) - v10_counts.get('Low', 0)
diff_med = pred_counts.get('Medium', 0) - v10_counts.get('Medium', 0)
diff_high = pred_counts.get('High', 0) - v10_counts.get('High', 0)

print(f"\nDifference:")
print(f"  Low: {diff_low:+d}")
print(f"  Medium: {diff_med:+d}")
print(f"  High: {diff_high:+d}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': v10_predictions})
filename = 'submissions/submission_v24_starter_aligned.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V24 COMPLETE - STARTER-ALIGNED")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: V10 ensemble with CORRECT profit_margin formula")
print(f"\nKEY FIX:")
print(f"  OLD: profit_margin = log1p(turnover / expenses)")
print(f"  NEW: profit_margin = (income - expenses) / income")
print(f"\nExpected: If this was the missing piece, should beat 0.892")
print(f"{'='*80}")
