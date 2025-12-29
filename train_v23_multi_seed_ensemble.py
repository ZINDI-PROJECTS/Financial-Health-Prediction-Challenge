"""
V23: MULTI-SEED ENSEMBLE

Strategy:
- Train V10 approach with 5 different random seeds
- Average predictions across all seeds
- Reduces variance, increases stability

Hypothesis: V10 might be seed-dependent
Different seeds → slightly different models → averaging reduces overfitting

This is NOT CV-based - just variance reduction through ensemble
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

print("="*80)
print("V23: MULTI-SEED ENSEMBLE")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/3] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# TRAIN WITH MULTIPLE SEEDS
# ============================================
print("\n[2/3] Training V10 approach with 5 different seeds...")

SEEDS = [42, 123, 456, 789, 1337]
class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

test_probas_all_seeds = []

for seed_idx, seed in enumerate(SEEDS, 1):
    print(f"\n{'='*60}")
    print(f"SEED {seed_idx}/5: {seed}")
    print(f"{'='*60}")

    np.random.seed(seed)

    # ============================================
    # V2 MODELS (NO SMOTE)
    # ============================================
    print("  [V2] Training CatBoost...", end=' ')
    v2_cat = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=seed,
        verbose=False
    )
    v2_cat.fit(X, y_encoded, cat_features=categorical_cols, verbose=False)
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
        random_state=seed,
        verbose=-1
    )
    v2_lgb.fit(X_lgb, y_encoded)
    v2_lgb_proba = v2_lgb.predict_proba(X_test_lgb)
    print("Done")

    v2_proba = (v2_cat_proba + v2_lgb_proba) / 2

    # ============================================
    # V9 MODELS (WITH SMOTE)
    # ============================================
    print("  [V9] Encoding for SMOTE...", end=' ')
    X_smote = X.copy()
    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]])
        le.fit(combined)
        X_smote[col] = le.transform(X[col])
    print("Done")

    print("  [V9] Applying SMOTE...", end=' ')
    sampling_strategy = {0: (y_encoded == 0).sum(), 1: (y_encoded == 1).sum(), 2: 1000}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed, k_neighbors=5)
    X_smote_resampled, y_smote = smote.fit_resample(X_smote, y_encoded)
    print("Done")

    print("  [V9] Training CatBoost...", end=' ')
    v9_cat = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=seed,
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
        random_state=seed,
        verbose=-1
    )
    v9_lgb.fit(X_smote_resampled, y_smote)
    v9_lgb_proba = v9_lgb.predict_proba(X_test_smote)
    print("Done")

    v9_proba = (v9_cat_proba + v9_lgb_proba) / 2

    # ============================================
    # BLEND V2 + V9 (50/50)
    # ============================================
    v10_proba = (v2_proba + v9_proba) / 2
    test_probas_all_seeds.append(v10_proba)

    # Show distribution for this seed
    seed_pred = np.argmax(v10_proba, axis=1)
    seed_predictions = [reverse_mapping[p] for p in seed_pred]
    seed_counts = pd.Series(seed_predictions).value_counts()

    print(f"  Predictions: L={seed_counts.get('Low', 0)}, "
          f"M={seed_counts.get('Medium', 0)}, "
          f"H={seed_counts.get('High', 0)}")

# ============================================
# ENSEMBLE ACROSS SEEDS
# ============================================
print("\n[3/3] Ensembling predictions across all seeds...")

# Average probabilities across all seeds
ensemble_proba = np.mean(test_probas_all_seeds, axis=0)
ensemble_pred = np.argmax(ensemble_proba, axis=1)
ensemble_predictions = [reverse_mapping[p] for p in ensemble_pred]

# Distribution
pred_counts = pd.Series(ensemble_predictions).value_counts()
print(f"\nV23 Multi-Seed Ensemble distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(ensemble_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V10
print(f"\nComparison to V10 (single seed=42):")
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v10_counts = v10_sub['Target'].value_counts()
print(f"  V10: Low={v10_counts.get('Low', 0)}, Med={v10_counts.get('Medium', 0)}, High={v10_counts.get('High', 0)}")
print(f"  V23: Low={pred_counts.get('Low', 0)}, Med={pred_counts.get('Medium', 0)}, High={pred_counts.get('High', 0)}")

# Difference analysis
diff_low = pred_counts.get('Low', 0) - v10_counts.get('Low', 0)
diff_med = pred_counts.get('Medium', 0) - v10_counts.get('Medium', 0)
diff_high = pred_counts.get('High', 0) - v10_counts.get('High', 0)

print(f"\nDifference from V10:")
print(f"  Low: {diff_low:+d}")
print(f"  Medium: {diff_med:+d}")
print(f"  High: {diff_high:+d}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': ensemble_predictions})
filename = 'submissions/submission_v23_multi_seed_ensemble.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V23 COMPLETE - MULTI-SEED ENSEMBLE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: V10 approach with 5 seeds (42, 123, 456, 789, 1337)")
print(f"Hypothesis: Averaging reduces variance, improves stability")
print(f"\nNOTE: This is NOT CV-based, just ensemble variance reduction")
print(f"{'='*80}")
