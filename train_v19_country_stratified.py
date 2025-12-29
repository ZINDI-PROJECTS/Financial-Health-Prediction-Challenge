"""
V19: COUNTRY-STRATIFIED ENSEMBLE (GPU)

Strategy:
- Train SEPARATE models for EACH country (Eswatini, Lesotho, Malawi, Zimbabwe)
- Hypothesis: Distribution shift might be country-specific
- Each country gets its own optimized CatBoost + LightGBM ensemble
- Blend country-specific predictions

Why this might work:
- V10 works well, but might be averaging across country-specific patterns
- Country interactions (V6 features) suggest country matters
- Each country might have different Low/Med/High distributions

Expected: If distribution shift is country-based, this could beat 0.892
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("V19: COUNTRY-STRATIFIED ENSEMBLE (GPU)")
print("="*80)

# ============================================
# CHECK GPU
# ============================================
print("\n[1/8] Checking GPU availability...")
import subprocess
try:
    gpu_info = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'])
    print(f"GPU detected: {gpu_info.decode().strip()}")
    USE_GPU = True
except:
    print("WARNING: No GPU detected, using CPU")
    USE_GPU = False

# ============================================
# LOAD DATA
# ============================================
print("\n[2/8] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# ANALYZE BY COUNTRY
# ============================================
print("\n[3/8] Analyzing country distributions...")
countries = ['Eswatini', 'Lesotho', 'Malawi', 'Zimbabwe']

for country in countries:
    train_mask = X['country'] == country
    test_mask = X_test['country'] == country

    train_count = train_mask.sum()
    test_count = test_mask.sum()

    if train_count > 0:
        country_dist = np.bincount(y_encoded[train_mask])
        print(f"\n  {country}:")
        print(f"    Train: {train_count:,} | Test: {test_count:,}")
        print(f"    Distribution: {country_dist} (Low/Med/High)")

# ============================================
# PREPARE DATA
# ============================================
print("\n[4/8] Preparing data for models...")

# CatBoost data (keep categoricals)
X_cat = X.copy()
X_test_cat = X_test.copy()

# LightGBM data (encode categoricals)
X_lgb = X.copy()
X_test_lgb = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_lgb[col] = le.transform(X[col])
    X_test_lgb[col] = le.transform(X_test[col])

# ============================================
# 5-FOLD CV (GLOBAL BASELINE)
# ============================================
print("\n[5/8] Training GLOBAL baseline for comparison...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_global = np.zeros(len(X))
test_proba_global = np.zeros((len(X_test), 3))

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ')

    X_train, X_val = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        task_type='GPU' if USE_GPU else 'CPU',
        devices='0' if USE_GPU else None,
        gpu_ram_part=0.5 if USE_GPU else None,
        class_weights=class_weights,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )

    cat_model.fit(X_train, y_train, cat_features=categorical_cols, verbose=False)

    oof_global[val_idx] = cat_model.predict(X_val).flatten()
    test_proba_global += cat_model.predict_proba(X_test_cat) / N_FOLDS

    print("Done")

global_f1 = f1_score(y_encoded, oof_global, average='macro')
print(f"\nGlobal CV Macro F1: {global_f1:.6f}")

# ============================================
# COUNTRY-STRATIFIED MODELS
# ============================================
print("\n[6/8] Training COUNTRY-STRATIFIED models...")

country_models = {}
test_predictions_by_country = {}

for country in countries:
    print(f"\n{'='*60}")
    print(f"TRAINING: {country.upper()}")
    print(f"{'='*60}")

    # Filter by country
    train_mask = X_cat['country'] == country
    test_mask = X_test_cat['country'] == country

    X_country = X_cat[train_mask].copy()
    y_country = y_encoded[train_mask]
    X_test_country = X_test_cat[test_mask].copy()

    print(f"  Train samples: {len(X_country):,}")
    print(f"  Test samples: {len(X_test_country):,}")
    print(f"  Distribution: {np.bincount(y_country)}")

    if len(X_country) < 50:
        print(f"  SKIPPING {country} - too few samples")
        continue

    # Train country-specific CatBoost
    print(f"  Training CatBoost for {country}...")

    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1:average=Macro',
        task_type='GPU' if USE_GPU else 'CPU',
        devices='0' if USE_GPU else None,
        gpu_ram_part=0.5 if USE_GPU else None,
        class_weights=class_weights,
        iterations=1500,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )

    cat_model.fit(X_country, y_country, cat_features=categorical_cols, verbose=False)

    # Train country-specific LightGBM
    print(f"  Training LightGBM for {country}...")

    X_country_lgb = X_lgb[train_mask].copy()
    X_test_country_lgb = X_test_lgb[test_mask].copy()

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        device='gpu' if USE_GPU else 'cpu',
        gpu_platform_id=0 if USE_GPU else None,
        gpu_device_id=0 if USE_GPU else None,
        class_weight=class_weights,
        n_estimators=1500,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )

    lgb_model.fit(X_country_lgb, y_country)

    # Predict for this country's test data
    cat_proba = cat_model.predict_proba(X_test_country)
    lgb_proba = lgb_model.predict_proba(X_test_country_lgb)

    # Blend
    country_proba = (cat_proba + lgb_proba) / 2
    country_pred = np.argmax(country_proba, axis=1)

    country_models[country] = {'catboost': cat_model, 'lightgbm': lgb_model}
    test_predictions_by_country[country] = {
        'predictions': country_pred,
        'indices': test_mask
    }

    print(f"  ✓ {country} models trained")

# ============================================
# COMBINE COUNTRY PREDICTIONS
# ============================================
print("\n[7/8] Combining country-specific predictions...")

final_predictions = np.zeros(len(X_test), dtype=int)

for country, data in test_predictions_by_country.items():
    mask = data['indices']
    preds = data['predictions']
    final_predictions[mask] = preds
    print(f"  {country}: {mask.sum():,} predictions")

test_predictions = [reverse_mapping[p] for p in final_predictions]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V10
print(f"\nComparison to V10:")
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v10_counts = v10_sub['Target'].value_counts()
print(f"  V10: Low={v10_counts.get('Low', 0)} ({v10_counts.get('Low', 0)/len(v10_sub)*100:.1f}%), "
      f"Med={v10_counts.get('Medium', 0)} ({v10_counts.get('Medium', 0)/len(v10_sub)*100:.1f}%), "
      f"High={v10_counts.get('High', 0)} ({v10_counts.get('High', 0)/len(v10_sub)*100:.1f}%)")
print(f"  V19: Low={pred_counts.get('Low', 0)} ({pred_counts.get('Low', 0)/len(test_predictions)*100:.1f}%), "
      f"Med={pred_counts.get('Medium', 0)} ({pred_counts.get('Medium', 0)/len(test_predictions)*100:.1f}%), "
      f"High={pred_counts.get('High', 0)} ({pred_counts.get('High', 0)/len(test_predictions)*100:.1f}%)")

# ============================================
# SAVE SUBMISSION
# ============================================
print("\n[8/8] Saving submission...")

submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v19_country_stratified.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V19 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: Country-Stratified Ensemble (4 country-specific models)")
print(f"Global CV F1: {global_f1:.6f} (for reference)")
print(f"GPU Used: {USE_GPU}")
print(f"\nHypothesis: If distribution shift is country-specific, this beats 0.892")
print(f"{'='*80}")
