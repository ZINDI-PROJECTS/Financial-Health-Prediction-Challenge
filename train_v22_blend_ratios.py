"""
V22: V10 BLEND RATIO EXPLORATION

Known LB scores (REAL data):
- V2 (CatBoost + LightGBM, no SMOTE): 0.883 LB
- V9 (CatBoost + LightGBM, with SMOTE): 0.888 LB
- V10 (50/50 blend of V2+V9): 0.892 LB

Observation: Blending improved performance!
V10's 50/50 > V9 alone > V2 alone

Hypothesis: Maybe a different ratio beats 50/50?
Test: 30/70, 40/60, 60/40, 70/30 (V2/V9 ratios)

Strategy: Generate all blend ratios, analyze distributions
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

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("V22: V10 BLEND RATIO EXPLORATION")
print("="*80)
print("\nKnown LB Performance:")
print("  V2 (no SMOTE):      0.883")
print("  V9 (SMOTE):         0.888")
print("  V10 (50/50 blend):  0.892 ⭐")
print("\nGoal: Find if different ratio beats 50/50")

# ============================================
# LOAD DATA
# ============================================
print("\n[1/4] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# TRAIN V2 MODELS (NO SMOTE)
# ============================================
print("\n[2/4] Training V2 models (no SMOTE)...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

print("  Training V2 CatBoost...", end=' ')
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
v2_cat.fit(X, y_encoded, cat_features=categorical_cols, verbose=False)
v2_cat_proba = v2_cat.predict_proba(X_test)
print("Done")

print("  Training V2 LightGBM...", end=' ')
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
v2_lgb.fit(X_lgb, y_encoded)
v2_lgb_proba = v2_lgb.predict_proba(X_test_lgb)
print("Done")

# V2 blended probabilities
v2_proba = (v2_cat_proba + v2_lgb_proba) / 2

# ============================================
# TRAIN V9 MODELS (WITH SMOTE)
# ============================================
print("\n[3/4] Training V9 models (SMOTE)...")

# Encode for SMOTE
X_smote = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_smote[col] = le.transform(X[col])

# Apply SMOTE
sampling_strategy = {0: (y_encoded == 0).sum(), 1: (y_encoded == 1).sum(), 2: 1000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_smote_resampled, y_smote = smote.fit_resample(X_smote, y_encoded)

print("  Training V9 CatBoost...", end=' ')
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

print("  Training V9 LightGBM...", end=' ')
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

# V9 blended probabilities
v9_proba = (v9_cat_proba + v9_lgb_proba) / 2

# ============================================
# TEST DIFFERENT BLEND RATIOS
# ============================================
print("\n[4/4] Testing different V2/V9 blend ratios...")

blend_ratios = [
    (0.3, 0.7, "30/70"),
    (0.4, 0.6, "40/60"),
    (0.5, 0.5, "50/50 (V10)"),
    (0.6, 0.4, "60/40"),
    (0.7, 0.3, "70/30")
]

results = []
v10_dist = None

print(f"\n{'Ratio (V2/V9)':<20} {'Low':<8} {'Medium':<8} {'High':<8} {'Notes'}")
print("="*70)

for v2_weight, v9_weight, label in blend_ratios:
    # Blend probabilities
    blend_proba = v2_weight * v2_proba + v9_weight * v9_proba
    blend_pred = np.argmax(blend_proba, axis=1)
    blend_predictions = [reverse_mapping[p] for p in blend_pred]

    # Distribution
    pred_counts = pd.Series(blend_predictions).value_counts()
    low_count = pred_counts.get('Low', 0)
    med_count = pred_counts.get('Medium', 0)
    high_count = pred_counts.get('High', 0)

    low_pct = low_count / len(blend_predictions) * 100
    med_pct = med_count / len(blend_predictions) * 100
    high_pct = high_count / len(blend_predictions) * 100

    # Save V10 for comparison
    if v2_weight == 0.5:
        v10_dist = (low_count, med_count, high_count)

    # Determine notes
    notes = ""
    if label == "50/50 (V10)":
        notes = "← BASELINE (0.892 LB)"
    elif high_count > v10_dist[2] if v10_dist else high_count > 94:
        notes = "More High predictions"
    elif high_count < v10_dist[2] if v10_dist else high_count < 94:
        notes = "Fewer High predictions"

    print(f"{label:<20} {low_count:<8} {med_count:<8} {high_count:<8} {notes}")

    results.append({
        'ratio': label,
        'v2_weight': v2_weight,
        'v9_weight': v9_weight,
        'low': low_count,
        'medium': med_count,
        'high': high_count,
        'predictions': blend_predictions
    })

# ============================================
# SAVE TOP CANDIDATES
# ============================================
print("\n" + "="*80)
print("SAVING CANDIDATE SUBMISSIONS")
print("="*80)

for i, result in enumerate(results):
    # Save each blend ratio
    submission = pd.DataFrame({'ID': test_ids, 'Target': result['predictions']})

    ratio_clean = result['ratio'].replace('/', '_').replace(' ', '_').replace('(', '').replace(')', '')
    filename = f"submissions/submission_v22_{ratio_clean}.csv"
    submission.to_csv(filename, index=False)

    print(f"  Saved: {filename}")
    print(f"    Ratio: {result['ratio']}")
    print(f"    Distribution: L={result['low']}, M={result['medium']}, H={result['high']}")

print("\n" + "="*80)
print("V22 COMPLETE - BLEND RATIO EXPLORATION")
print("="*80)
print("\nKEY INSIGHT:")
print("  We now have 5 different blends of V2 (0.883) + V9 (0.888)")
print("  V10's 50/50 blend scored 0.892 LB")
print("  Question: Does a different ratio beat 50/50?")
print("\nNEXT STEP:")
print("  Analyze distributions to identify best candidate")
print("  Look for meaningful differences from V10")
print("="*80)
