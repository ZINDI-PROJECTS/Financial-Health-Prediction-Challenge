"""
V14: PROBABILITY CALIBRATION - ISOTONIC + TEMPERATURE SCALING

Problem:
- Model probabilities may not be well-calibrated
- A predicted 80% confidence might actually be 85% or 75% in reality
- Miscalibration leads to suboptimal threshold-based predictions

Solution: Calibrate probabilities using OOF data
- Isotonic Regression: Non-parametric, fits monotonic function
- Temperature Scaling: Divides logits by temperature T
- Apply to test probabilities for better predictions

Expected: +0.002-0.006 gain from proper calibration
Target: 0.894-0.898 LB
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

def optimize_thresholds(y_true, y_proba):
    """Find optimal thresholds for macro-F1"""
    def objective(thresholds):
        predictions = np.argmax(y_proba - thresholds, axis=1)
        return -f1_score(y_true, predictions, average='macro')
    result = minimize(objective, np.zeros(3), method='Nelder-Mead', options={'maxiter': 500})
    return result.x

def predict_with_thresholds(y_proba, thresholds):
    return np.argmax(y_proba - thresholds, axis=1)

def temperature_scaling(proba, temperature):
    """Apply temperature scaling to probabilities"""
    # Convert to logits
    epsilon = 1e-10
    proba_clipped = np.clip(proba, epsilon, 1 - epsilon)
    logits = np.log(proba_clipped)

    # Scale by temperature
    scaled_logits = logits / temperature

    # Convert back to probabilities (softmax)
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    calibrated_proba = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    return calibrated_proba

def optimize_temperature(y_true, y_proba):
    """Find optimal temperature for temperature scaling"""
    def objective(temp):
        calibrated = temperature_scaling(y_proba, temp[0])
        return log_loss(y_true, calibrated)

    result = minimize(objective, [1.0], bounds=[(0.1, 10.0)], method='L-BFGS-B')
    return result.x[0]

print("="*80)
print("V14: PROBABILITY CALIBRATION - ISOTONIC + TEMPERATURE")
print("="*80)
print(f"Current: V13 (optimized blend) → 0.892 LB")
print(f"Target:  V14 (calibrated) → 0.894-0.898 LB")
print(f"Leader:  0.906 LB")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# 2. PREPARE DATA
# ============================================
print("\n[2/6] Preparing V2 (no SMOTE) and V9 (SMOTE) data...")

# V2 data
X_v2 = X.copy()
X_test_v2 = X_test.copy()

X_v2_encoded = X_v2.copy()
X_test_v2_encoded = X_test_v2.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_v2[col], X_test_v2[col]])
    le.fit(combined)
    X_v2_encoded[col] = le.transform(X_v2[col])
    X_test_v2_encoded[col] = le.transform(X_test_v2[col])

# V9 data with SMOTE
X_v9 = X.copy()
X_test_v9 = X_test.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_v9[col] = le.transform(X[col])
    X_test_v9[col] = le.transform(X_test[col])

sampling_strategy = {0: (y_encoded == 0).sum(), 1: (y_encoded == 1).sum(), 2: 1000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_v9_resampled, y_v9 = smote.fit_resample(X_v9, y_encoded)
X_v9_resampled = pd.DataFrame(X_v9_resampled, columns=X_v9.columns)

# ============================================
# 3. GENERATE OOF PROBABILITIES
# ============================================
print("\n[3/6] Generating OOF probabilities for calibration...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

oof_v2_proba = np.zeros((len(X_v2), 3))
oof_v9_proba = np.zeros((len(X_v2), 3))
oof_labels = np.zeros(len(X_v2), dtype=int)

print(f"Training models with 5-fold CV...")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_v2, y_encoded), 1):
    print(f"  Fold {fold_idx}/5...", end=' ')

    oof_labels[val_idx] = y_encoded.iloc[val_idx].values

    # V2 models
    X_v2_train, X_v2_val = X_v2.iloc[train_idx], X_v2.iloc[val_idx]
    X_v2_train_enc, X_v2_val_enc = X_v2_encoded.iloc[train_idx], X_v2_encoded.iloc[val_idx]
    y_v2_train = y_encoded.iloc[train_idx]

    v2_cat = CatBoostClassifier(
        loss_function='MultiClass', class_weights=class_weights,
        iterations=1000, learning_rate=0.05, depth=6,
        random_state=RANDOM_STATE, verbose=False
    )
    v2_cat.fit(X_v2_train, y_v2_train, cat_features=categorical_cols, verbose=False)

    v2_lgb = LGBMClassifier(
        objective='multiclass', num_class=3, class_weight=class_weights,
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_STATE, verbose=-1
    )
    v2_lgb.fit(X_v2_train_enc, y_v2_train)

    v2_blend = (v2_cat.predict_proba(X_v2_val) + v2_lgb.predict_proba(X_v2_val_enc)) / 2
    oof_v2_proba[val_idx] = v2_blend

    # V9 models
    X_v9_fold_train = X_v9.iloc[train_idx]
    y_v9_fold_train = y_encoded.iloc[train_idx]
    X_v9_fold_val = X_v9.iloc[val_idx]

    smote_fold = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_v9_fold_train_resampled, y_v9_fold_train_resampled = smote_fold.fit_resample(X_v9_fold_train, y_v9_fold_train)
    X_v9_fold_train_resampled = pd.DataFrame(X_v9_fold_train_resampled, columns=X_v9.columns)

    v9_cat = CatBoostClassifier(
        loss_function='MultiClass', class_weights=class_weights,
        iterations=1000, learning_rate=0.05, depth=6,
        random_state=RANDOM_STATE, verbose=False
    )
    v9_cat.fit(X_v9_fold_train_resampled, y_v9_fold_train_resampled, verbose=False)

    v9_lgb = LGBMClassifier(
        objective='multiclass', num_class=3, class_weight=class_weights,
        n_estimators=1000, learning_rate=0.05, max_depth=6,
        random_state=RANDOM_STATE, verbose=-1
    )
    v9_lgb.fit(X_v9_fold_train_resampled, y_v9_fold_train_resampled)

    v9_blend = (v9_cat.predict_proba(X_v9_fold_val) + v9_lgb.predict_proba(X_v9_fold_val)) / 2
    oof_v9_proba[val_idx] = v9_blend

    print("Done")

# Use V13's optimal blend weight
OPTIMAL_ALPHA = 0.45
oof_blended_proba = OPTIMAL_ALPHA * oof_v9_proba + (1 - OPTIMAL_ALPHA) * oof_v2_proba

print(f"\nOOF probabilities generated: {oof_blended_proba.shape}")

# ============================================
# 4. CALIBRATE PROBABILITIES
# ============================================
print("\n[4/6] Calibrating probabilities...")

print("\nMethod 1: Temperature Scaling")
optimal_temp = optimize_temperature(oof_labels, oof_blended_proba)
oof_temp_calibrated = temperature_scaling(oof_blended_proba, optimal_temp)

# Evaluate
thresholds_uncalib = optimize_thresholds(oof_labels, oof_blended_proba)
pred_uncalib = predict_with_thresholds(oof_blended_proba, thresholds_uncalib)
f1_uncalib = f1_score(oof_labels, pred_uncalib, average='macro')

thresholds_temp = optimize_thresholds(oof_labels, oof_temp_calibrated)
pred_temp = predict_with_thresholds(oof_temp_calibrated, thresholds_temp)
f1_temp = f1_score(oof_labels, pred_temp, average='macro')

print(f"  Optimal temperature: {optimal_temp:.3f}")
print(f"  Uncalibrated F1: {f1_uncalib:.6f}")
print(f"  Temp-calibrated F1: {f1_temp:.6f}")
print(f"  Gain: {f1_temp - f1_uncalib:+.6f}")

print("\nMethod 2: Isotonic Regression (per-class)")
isotonic_calibrators = []
oof_iso_calibrated = oof_blended_proba.copy()

for class_idx in range(3):
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_blended_proba[:, class_idx], (oof_labels == class_idx).astype(int))
    oof_iso_calibrated[:, class_idx] = iso.predict(oof_blended_proba[:, class_idx])
    isotonic_calibrators.append(iso)

# Normalize probabilities
oof_iso_calibrated = oof_iso_calibrated / oof_iso_calibrated.sum(axis=1, keepdims=True)

thresholds_iso = optimize_thresholds(oof_labels, oof_iso_calibrated)
pred_iso = predict_with_thresholds(oof_iso_calibrated, thresholds_iso)
f1_iso = f1_score(oof_labels, pred_iso, average='macro')

print(f"  Iso-calibrated F1: {f1_iso:.6f}")
print(f"  Gain: {f1_iso - f1_uncalib:+.6f}")

# Choose best method
if f1_temp >= f1_iso:
    print(f"\n✓ Using Temperature Scaling (T={optimal_temp:.3f})")
    best_method = 'temperature'
    best_f1 = f1_temp
    calibrated_proba = oof_temp_calibrated
    final_thresholds = thresholds_temp
else:
    print(f"\n✓ Using Isotonic Regression")
    best_method = 'isotonic'
    best_f1 = f1_iso
    calibrated_proba = oof_iso_calibrated
    final_thresholds = thresholds_iso

print(f"\n{'='*80}")
print("CALIBRATION RESULTS")
print(f"{'='*80}")
print(f"V13 Uncalibrated: {f1_uncalib:.6f} OOF F1")
print(f"V14 Calibrated:   {best_f1:.6f} OOF F1")
print(f"Gain:             {best_f1 - f1_uncalib:+.6f}")

# ============================================
# 5. TRAIN FINAL MODELS
# ============================================
print(f"\n[5/6] Training final models...")

# Train V2
print("  Training V2 models...")
final_v2_cat = CatBoostClassifier(
    loss_function='MultiClass', class_weights=class_weights,
    iterations=1000, learning_rate=0.05, depth=6,
    random_state=RANDOM_STATE, verbose=False
)
final_v2_cat.fit(X_v2, y_encoded, cat_features=categorical_cols, verbose=False)

final_v2_lgb = LGBMClassifier(
    objective='multiclass', num_class=3, class_weight=class_weights,
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=RANDOM_STATE, verbose=-1
)
final_v2_lgb.fit(X_v2_encoded, y_encoded)

# Train V9
print("  Training V9 models...")
final_v9_cat = CatBoostClassifier(
    loss_function='MultiClass', class_weights=class_weights,
    iterations=1000, learning_rate=0.05, depth=6,
    random_state=RANDOM_STATE, verbose=False
)
final_v9_cat.fit(X_v9_resampled, y_v9, verbose=False)

final_v9_lgb = LGBMClassifier(
    objective='multiclass', num_class=3, class_weight=class_weights,
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    random_state=RANDOM_STATE, verbose=-1
)
final_v9_lgb.fit(X_v9_resampled, y_v9)

# ============================================
# 6. PREDICT WITH CALIBRATION
# ============================================
print(f"\n[6/6] Generating calibrated predictions...")

# Get test probabilities
v2_test_proba = (final_v2_cat.predict_proba(X_test_v2) + final_v2_lgb.predict_proba(X_test_v2_encoded)) / 2
v9_test_proba = (final_v9_cat.predict_proba(X_test_v9) + final_v9_lgb.predict_proba(X_test_v9)) / 2

# Blend with optimal weight
test_blended_proba = OPTIMAL_ALPHA * v9_test_proba + (1 - OPTIMAL_ALPHA) * v2_test_proba

# Apply calibration
if best_method == 'temperature':
    test_calibrated_proba = temperature_scaling(test_blended_proba, optimal_temp)
else:
    test_calibrated_proba = test_blended_proba.copy()
    for class_idx in range(3):
        test_calibrated_proba[:, class_idx] = isotonic_calibrators[class_idx].predict(test_blended_proba[:, class_idx])
    test_calibrated_proba = test_calibrated_proba / test_calibrated_proba.sum(axis=1, keepdims=True)

# Predict
test_pred_encoded = predict_with_thresholds(test_calibrated_proba, final_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nV14 Test predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare
v13_sub = pd.read_csv('submissions/submission_v13_optimized_blend_weights.csv')
print(f"\nComparison to V13:")
print(f"  V13 High: {len(v13_sub[v13_sub['Target']=='High'])}")
print(f"  V14 High: {pred_counts.get('High', 0)}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v14_probability_calibration.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V14 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: {best_method.title()} calibration")
print(f"OOF Macro-F1: {best_f1:.6f}")
print(f"Gain over V13: {best_f1 - f1_uncalib:+.6f}")
print(f"\nExpected LB: 0.894-0.898 (Leader: 0.906)")
print(f"{'='*80}")
