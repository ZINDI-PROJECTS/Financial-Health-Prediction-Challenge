"""
V20: V10 PROBABILITY CALIBRATION

Strategy:
- Take V10's exact approach (V2 + V9 blend)
- BUT: Calibrate probabilities using Isotonic Regression on OOF predictions
- This adjusts decision boundaries without changing the model
- Low-risk, proven technique to squeeze extra performance

Why this might work:
- V10 already works (0.892), we're just optimizing boundaries
- Calibration fixes probability miscalibration issues
- Can gain 0.001-0.005 in F1 score

Expected: 0.893-0.897 (small but safe improvement)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
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
print("V20: V10 PROBABILITY CALIBRATION")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/5] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# REPLICATE V10 WITH OOF PREDICTIONS
# ============================================
print("\n[2/5] Replicating V10 approach with OOF tracking...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Store OOF probabilities for calibration
oof_probas_v2 = np.zeros((len(X), 3))
oof_probas_v9 = np.zeros((len(X), 3))

# Store test probabilities
test_probas_v2 = np.zeros((len(X_test), 3))
test_probas_v9 = np.zeros((len(X_test), 3))

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

print("\n  Training V2 models (no SMOTE) with OOF...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"    Fold {fold}/{N_FOLDS}...", end=' ')

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # V2 CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )
    cat_model.fit(X_train, y_train, cat_features=categorical_cols, verbose=False)

    # V2 LightGBM (encode categoricals)
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]])
        le.fit(combined)
        X_train_lgb[col] = le.transform(X_train[col])
        X_val_lgb[col] = le.transform(X_val[col])
        X_test_lgb[col] = le.transform(X_test[col])

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train)

    # OOF predictions
    cat_proba = cat_model.predict_proba(X_val)
    lgb_proba = lgb_model.predict_proba(X_val_lgb)
    oof_probas_v2[val_idx] = (cat_proba + lgb_proba) / 2

    # Test predictions
    cat_test_proba = cat_model.predict_proba(X_test)
    lgb_test_proba = lgb_model.predict_proba(X_test_lgb)
    test_probas_v2 += (cat_test_proba + lgb_test_proba) / 2 / N_FOLDS

    print("Done")

print("\n  Training V9 models (SMOTE) with OOF...")

# Prepare SMOTE data
X_smote = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_smote[col] = le.transform(X[col])

sampling_strategy = {0: (y_encoded == 0).sum(), 1: (y_encoded == 1).sum(), 2: 1000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_smote_resampled, y_smote = smote.fit_resample(X_smote, y_encoded)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"    Fold {fold}/{N_FOLDS}...", end=' ')

    X_val = X_smote.iloc[val_idx]
    X_test_smote = X_test.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]])
        le.fit(combined)
        X_test_smote[col] = le.transform(X_test[col])

    # V9 CatBoost (on SMOTE data)
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )
    cat_model.fit(X_smote_resampled, y_smote, verbose=False)

    # V9 LightGBM
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_smote_resampled, y_smote)

    # OOF predictions
    cat_proba = cat_model.predict_proba(X_val)
    lgb_proba = lgb_model.predict_proba(X_val)
    oof_probas_v9[val_idx] = (cat_proba + lgb_proba) / 2

    # Test predictions
    cat_test_proba = cat_model.predict_proba(X_test_smote)
    lgb_test_proba = lgb_model.predict_proba(X_test_smote)
    test_probas_v9 += (cat_test_proba + lgb_test_proba) / 2 / N_FOLDS

    print("Done")

# ============================================
# CALIBRATE PROBABILITIES
# ============================================
print("\n[3/5] Calibrating probabilities using Isotonic Regression...")

# Blend V2 + V9 OOF probabilities
oof_probas_blend = (oof_probas_v2 + oof_probas_v9) / 2

# Calibrate each class probability separately
calibrators = []
oof_probas_calibrated = np.zeros_like(oof_probas_blend)

for class_idx in range(3):
    print(f"  Calibrating class {reverse_mapping[class_idx]}...", end=' ')

    # Isotonic regression calibrator
    iso = IsotonicRegression(out_of_bounds='clip')

    # Fit on OOF probabilities
    y_binary = (y_encoded == class_idx).astype(int)
    iso.fit(oof_probas_blend[:, class_idx], y_binary)

    # Transform OOF probabilities
    oof_probas_calibrated[:, class_idx] = iso.transform(oof_probas_blend[:, class_idx])

    calibrators.append(iso)
    print("Done")

# Normalize calibrated probabilities
oof_probas_calibrated = oof_probas_calibrated / oof_probas_calibrated.sum(axis=1, keepdims=True)

# ============================================
# EVALUATE CALIBRATION
# ============================================
print("\n[4/5] Evaluating calibration...")

# Before calibration
oof_pred_before = np.argmax(oof_probas_blend, axis=1)
f1_before = f1_score(y_encoded, oof_pred_before, average='macro')

# After calibration
oof_pred_after = np.argmax(oof_probas_calibrated, axis=1)
f1_after = f1_score(y_encoded, oof_pred_after, average='macro')

print(f"\n  OOF F1 Before Calibration: {f1_before:.6f}")
print(f"  OOF F1 After Calibration:  {f1_after:.6f}")
print(f"  Gain: {f1_after - f1_before:+.6f}")

print(f"\n  Before Calibration:")
print(f"    Per-class F1: {f1_score(y_encoded, oof_pred_before, average=None)}")

print(f"\n  After Calibration:")
print(f"    Per-class F1: {f1_score(y_encoded, oof_pred_after, average=None)}")

# ============================================
# APPLY CALIBRATION TO TEST SET
# ============================================
print("\n[5/5] Applying calibration to test set...")

# Blend V2 + V9 test probabilities
test_probas_blend = (test_probas_v2 + test_probas_v9) / 2

# Apply calibration
test_probas_calibrated = np.zeros_like(test_probas_blend)
for class_idx in range(3):
    test_probas_calibrated[:, class_idx] = calibrators[class_idx].transform(test_probas_blend[:, class_idx])

# Normalize
test_probas_calibrated = test_probas_calibrated / test_probas_calibrated.sum(axis=1, keepdims=True)

# Make predictions
test_pred_encoded = np.argmax(test_probas_calibrated, axis=1)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V10
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v10_counts = v10_sub['Target'].value_counts()
print(f"\nComparison to V10:")
print(f"  V10: Low={v10_counts.get('Low', 0)} ({v10_counts.get('Low', 0)/len(v10_sub)*100:.1f}%), "
      f"Med={v10_counts.get('Medium', 0)} ({v10_counts.get('Medium', 0)/len(v10_sub)*100:.1f}%), "
      f"High={v10_counts.get('High', 0)} ({v10_counts.get('High', 0)/len(v10_sub)*100:.1f}%)")
print(f"  V20: Low={pred_counts.get('Low', 0)} ({pred_counts.get('Low', 0)/len(test_predictions)*100:.1f}%), "
      f"Med={pred_counts.get('Medium', 0)} ({pred_counts.get('Medium', 0)/len(test_predictions)*100:.1f}%), "
      f"High={pred_counts.get('High', 0)} ({pred_counts.get('High', 0)/len(test_predictions)*100:.1f}%)")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v20_v10_calibrated.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V20 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: V10 + Isotonic Probability Calibration")
print(f"OOF F1 Gain: {f1_after - f1_before:+.6f}")
print(f"Expected LB: 0.893-0.897 (safe, small improvement)")
print(f"{'='*80}")
