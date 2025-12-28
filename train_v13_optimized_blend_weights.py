"""
V13: OPTIMIZED BLEND WEIGHTS - OPTUNA SEARCH

Current V10: V2 + V9 with 50/50 blend → 0.892 LB
Problem: 50/50 is arbitrary, not optimal
Goal: Find mathematically optimal blend weight α

Strategy:
P(V13) = α * P(V9) + (1-α) * P(V2)

Where:
- P(V2) = No SMOTE, conservative predictions, good generalization
- P(V9) = SMOTE, better High class detection
- α = optimal weight found by Optuna (search 0.3-0.7)

Expected: +0.003-0.008 gain from optimal weighting
Target: 0.895-0.900 LB
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from imblearn.over_sampling import SMOTE
import optuna
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

print("="*80)
print("V13: OPTIMIZED BLEND WEIGHTS - OPTUNA SEARCH")
print("="*80)
print(f"Current: V10 (50/50 blend) → 0.892 LB")
print(f"Target:  V13 (optimal blend) → 0.895-0.900 LB")
print(f"Leader:  0.906 LB")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/5] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# 2. PREPARE DATA FOR BOTH APPROACHES
# ============================================
print("\n[2/5] Preparing V2 (no SMOTE) and V9 (SMOTE) data...")

# V2: Original data with categoricals
X_v2 = X.copy()
X_test_v2 = X_test.copy()

# Encode categoricals for V2 (for LightGBM)
X_v2_encoded = X_v2.copy()
X_test_v2_encoded = X_test_v2.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_v2[col], X_test_v2[col]])
    le.fit(combined)
    X_v2_encoded[col] = le.transform(X_v2[col])
    X_test_v2_encoded[col] = le.transform(X_test_v2[col])

# V9: SMOTE data (all encoded)
X_v9 = X.copy()
X_test_v9 = X_test.copy()
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_v9[col] = le.transform(X[col])
    X_test_v9[col] = le.transform(X_test[col])

# Apply SMOTE
sampling_strategy = {0: (y_encoded == 0).sum(), 1: (y_encoded == 1).sum(), 2: 1000}
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
X_v9_resampled, y_v9 = smote.fit_resample(X_v9, y_encoded)
X_v9_resampled = pd.DataFrame(X_v9_resampled, columns=X_v9.columns)

print(f"  V2 samples: {len(X_v2):,}")
print(f"  V9 samples (after SMOTE): {len(X_v9_resampled):,}")

# ============================================
# 3. GENERATE OOF PROBABILITIES
# ============================================
print("\n[3/5] Generating OOF probabilities with 5-fold CV...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# OOF storage
oof_v2_proba = np.zeros((len(X_v2), 3))
oof_v9_proba = np.zeros((len(X_v2), 3))  # Same size as original data
oof_indices = []
oof_labels = []

print(f"\n{'='*80}")
print("GENERATING OOF PROBABILITIES FOR V2 (NO SMOTE) AND V9 (SMOTE)")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_v2, y_encoded), 1):
    print(f"\nFold {fold_idx}/5:")

    oof_indices.extend(val_idx)
    oof_labels.extend(y_encoded.iloc[val_idx])

    # ============================================
    # V2: Train on original data (no SMOTE)
    # ============================================
    X_v2_train, X_v2_val = X_v2.iloc[train_idx], X_v2.iloc[val_idx]
    X_v2_train_enc, X_v2_val_enc = X_v2_encoded.iloc[train_idx], X_v2_encoded.iloc[val_idx]
    y_v2_train, y_v2_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # V2 CatBoost
    print("  V2: Training CatBoost...", end=' ')
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
    v2_cat.fit(X_v2_train, y_v2_train, cat_features=categorical_cols, verbose=False)
    v2_cat_proba = v2_cat.predict_proba(X_v2_val)
    print("Done", end=', ')

    # V2 LightGBM
    print("LightGBM...", end=' ')
    v2_lgb = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    v2_lgb.fit(X_v2_train_enc, y_v2_train)
    v2_lgb_proba = v2_lgb.predict_proba(X_v2_val_enc)
    print("Done")

    # V2 blend
    v2_blend_proba = (v2_cat_proba + v2_lgb_proba) / 2
    oof_v2_proba[val_idx] = v2_blend_proba

    # ============================================
    # V9: Train on SMOTE data
    # ============================================
    # For V9, we need to apply SMOTE to the training fold
    X_v9_fold_train = X_v9.iloc[train_idx]
    y_v9_fold_train = y_encoded.iloc[train_idx]
    X_v9_fold_val = X_v9.iloc[val_idx]

    # Apply SMOTE to training fold
    smote_fold = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=5)
    X_v9_fold_train_resampled, y_v9_fold_train_resampled = smote_fold.fit_resample(X_v9_fold_train, y_v9_fold_train)
    X_v9_fold_train_resampled = pd.DataFrame(X_v9_fold_train_resampled, columns=X_v9.columns)

    # V9 CatBoost
    print("  V9: Training CatBoost...", end=' ')
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
    v9_cat.fit(X_v9_fold_train_resampled, y_v9_fold_train_resampled, verbose=False)
    v9_cat_proba = v9_cat.predict_proba(X_v9_fold_val)
    print("Done", end=', ')

    # V9 LightGBM
    print("LightGBM...", end=' ')
    v9_lgb = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    v9_lgb.fit(X_v9_fold_train_resampled, y_v9_fold_train_resampled)
    v9_lgb_proba = v9_lgb.predict_proba(X_v9_fold_val)
    print("Done")

    # V9 blend
    v9_blend_proba = (v9_cat_proba + v9_lgb_proba) / 2
    oof_v9_proba[val_idx] = v9_blend_proba

# Convert to arrays in correct order
oof_indices = np.array(oof_indices)
oof_labels = np.array(oof_labels)

print(f"\n{'='*80}")
print("OOF PROBABILITIES GENERATED")
print(f"{'='*80}")
print(f"OOF samples: {len(oof_labels):,}")
print(f"V2 OOF shape: {oof_v2_proba.shape}")
print(f"V9 OOF shape: {oof_v9_proba.shape}")

# ============================================
# 4. OPTUNA OPTIMIZATION
# ============================================
print("\n[4/5] Optimizing blend weight α with Optuna...")

def objective(trial):
    """Optuna objective: find optimal blend weight α"""
    alpha = trial.suggest_float('alpha', 0.3, 0.7, step=0.01)

    # Blend: α * V9 + (1-α) * V2
    blended_proba = alpha * oof_v9_proba + (1 - alpha) * oof_v2_proba

    # Optimize thresholds
    thresholds = optimize_thresholds(oof_labels, blended_proba)

    # Predict
    predictions = predict_with_thresholds(blended_proba, thresholds)

    # Calculate F1
    f1 = f1_score(oof_labels, predictions, average='macro')

    return f1

# Run Optuna study
print(f"\nRunning Optuna study (50 trials)...")
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
study.optimize(objective, n_trials=50, timeout=600, show_progress_bar=True)

# Best parameters
best_alpha = study.best_params['alpha']
best_f1 = study.best_value

print(f"\n{'='*80}")
print("OPTUNA OPTIMIZATION RESULTS")
print(f"{'='*80}")
print(f"Best α (blend weight): {best_alpha:.3f}")
print(f"Best OOF Macro-F1: {best_f1:.6f}")
print(f"\nFormula: P(V13) = {best_alpha:.3f} * P(V9) + {1-best_alpha:.3f} * P(V2)")
print(f"\nInterpretation:")
if best_alpha > 0.55:
    print(f"  → SMOTE (V9) weighted MORE ({best_alpha*100:.1f}%)")
    print(f"  → V9's High class detection is more valuable")
elif best_alpha < 0.45:
    print(f"  → No-SMOTE (V2) weighted MORE ({(1-best_alpha)*100:.1f}%)")
    print(f"  → V2's generalization is more valuable")
else:
    print(f"  → Nearly balanced blend (V10's 50/50 was close to optimal)")

# Compare to V10 (50/50 blend)
v10_blended_proba = 0.5 * oof_v9_proba + 0.5 * oof_v2_proba
v10_thresholds = optimize_thresholds(oof_labels, v10_blended_proba)
v10_pred = predict_with_thresholds(v10_blended_proba, v10_thresholds)
v10_f1 = f1_score(oof_labels, v10_pred, average='macro')

print(f"\nComparison:")
print(f"  V10 (α=0.50):          {v10_f1:.6f} OOF F1")
print(f"  V13 (α={best_alpha:.2f}):        {best_f1:.6f} OOF F1")
print(f"  Gain:                  {best_f1 - v10_f1:+.6f}")

# ============================================
# 5. TRAIN FINAL MODEL WITH OPTIMAL WEIGHTS
# ============================================
print(f"\n[5/5] Training final models with optimal α={best_alpha:.3f}...")

# Train V2 models on full data
print("\n  Training V2 models (no SMOTE)...")
print("    CatBoost...", end=' ')
final_v2_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_v2_cat.fit(X_v2, y_encoded, cat_features=categorical_cols, verbose=False)
print("Done")

print("    LightGBM...", end=' ')
final_v2_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
final_v2_lgb.fit(X_v2_encoded, y_encoded)
print("Done")

# Train V9 models on SMOTE data
print("\n  Training V9 models (SMOTE)...")
print("    CatBoost...", end=' ')
final_v9_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_v9_cat.fit(X_v9_resampled, y_v9, verbose=False)
print("Done")

print("    LightGBM...", end=' ')
final_v9_lgb = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    class_weight=class_weights,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
final_v9_lgb.fit(X_v9_resampled, y_v9)
print("Done")

# Predict on test
print("\n  Generating test predictions...")
v2_cat_test_proba = final_v2_cat.predict_proba(X_test_v2)
v2_lgb_test_proba = final_v2_lgb.predict_proba(X_test_v2_encoded)
v2_test_proba = (v2_cat_test_proba + v2_lgb_test_proba) / 2

v9_cat_test_proba = final_v9_cat.predict_proba(X_test_v9)
v9_lgb_test_proba = final_v9_lgb.predict_proba(X_test_v9)
v9_test_proba = (v9_cat_test_proba + v9_lgb_test_proba) / 2

# Blend with optimal α
v13_test_proba = best_alpha * v9_test_proba + (1 - best_alpha) * v2_test_proba

# Use OOF-optimized thresholds
v13_blended_oof_proba = best_alpha * oof_v9_proba + (1 - best_alpha) * oof_v2_proba
final_thresholds = optimize_thresholds(oof_labels, v13_blended_oof_proba)

test_pred_encoded = predict_with_thresholds(v13_test_proba, final_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nV13 Test predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V10
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
print(f"\nComparison to V10:")
print(f"  V10 High predictions: {len(v10_sub[v10_sub['Target']=='High'])}")
print(f"  V13 High predictions: {pred_counts.get('High', 0)}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v13_optimized_blend_weights.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V13 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: Optuna-optimized blend (α={best_alpha:.3f})")
print(f"OOF Macro-F1: {best_f1:.6f}")
print(f"Gain over V10: {best_f1 - v10_f1:+.6f} OOF F1")
print(f"\nExpected LB: 0.895-0.900 (Leader: 0.906)")
print(f"{'='*80}")
