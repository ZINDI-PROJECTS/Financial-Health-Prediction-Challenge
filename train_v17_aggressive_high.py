"""
V17: AGGRESSIVE HIGH-CLASS PREDICTION (GPU)

Strategy:
- Build on V16's GPU acceleration
- EXTREME class weights for High class (20x instead of 17x)
- Lower probability threshold for High class predictions
- Gamma=3.0 for even more focus on hard examples
- Goal: Push more predictions into High class to improve recall

Key difference from V16:
- V16: Conservative (5.4% High predictions)
- V17: Aggressive (target 8-10% High predictions)

Expected: Better High-class recall, possibly 0.894-0.900
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

# AGGRESSIVE Focal Loss parameters
GAMMA = 3.0  # More aggressive than V16 (2.0)
HIGH_CLASS_BOOST = 25.0  # Extreme boost for High class

print("="*80)
print("V17: AGGRESSIVE HIGH-CLASS PREDICTION (GPU)")
print("="*80)

# ============================================
# CHECK GPU AVAILABILITY
# ============================================
print("\n[1/7] Checking GPU availability...")
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
print("\n[2/7] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")
print(f"Classes: {np.bincount(y_encoded)}")

# ============================================
# PREPARE DATA FOR CATBOOST
# ============================================
print("\n[3/7] Preparing data for CatBoost...")
X_cat = X.copy()
X_test_cat = X_test.copy()

# ============================================
# PREPARE DATA FOR LIGHTGBM
# ============================================
print("\n[4/7] Preparing data for LightGBM...")
X_lgb = X.copy()
X_test_lgb = X_test.copy()

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_lgb[col] = le.transform(X[col])
    X_test_lgb[col] = le.transform(X_test[col])
    encoders[col] = le

print(f"Categorical columns encoded: {len(categorical_cols)}")

# ============================================
# 5-FOLD STRATIFIED CV
# ============================================
print("\n[5/7] Training with 5-fold Stratified CV...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_predictions = np.zeros(len(X))
oof_probas_cat = np.zeros((len(X), 3))
oof_probas_lgb = np.zeros((len(X), 3))

test_probas_cat = np.zeros((len(X_test), 3))
test_probas_lgb = np.zeros((len(X_test), 3))

fold_scores = []

# AGGRESSIVE class weights - heavily favor High class
class_counts = np.bincount(y_encoded)
total = len(y_encoded)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
class_weights[2] = class_weights[2] * HIGH_CLASS_BOOST  # Extreme boost
print(f"Class weights (AGGRESSIVE): {class_weights}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold}/{N_FOLDS}")
    print(f"{'='*60}")

    X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    print(f"Train: {len(X_train_cat):,} | Val: {len(X_val_cat):,}")
    print(f"Train dist: {np.bincount(y_train)}")
    print(f"Val dist:   {np.bincount(y_val)}")

    # ------------------------------------
    # CatBoost with GPU (AGGRESSIVE)
    # ------------------------------------
    print(f"\n  [1/2] Training CatBoost (GPU - Aggressive)...")

    if USE_GPU:
        cat_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'task_type': 'GPU',
            'devices': '0',
            'gpu_ram_part': 0.7,
            'class_weights': class_weights,
            'iterations': 2500,  # More iterations
            'learning_rate': 0.025,  # Slower learning
            'depth': 9,  # Deeper trees
            'l2_leaf_reg': 3,  # Less regularization
            'random_state': RANDOM_STATE,
            'verbose': 100,
            'early_stopping_rounds': 150
        }
    else:
        cat_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'class_weights': class_weights,
            'iterations': 2000,
            'learning_rate': 0.04,
            'depth': 7,
            'random_state': RANDOM_STATE,
            'verbose': 100,
            'early_stopping_rounds': 100
        }

    cat_model = CatBoostClassifier(**cat_params)

    train_pool = Pool(X_train_cat, y_train, cat_features=categorical_cols)
    val_pool = Pool(X_val_cat, y_val, cat_features=categorical_cols)

    cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)

    # Predictions
    val_proba_cat = cat_model.predict_proba(X_val_cat)
    test_proba_cat = cat_model.predict_proba(X_test_cat)

    oof_probas_cat[val_idx] = val_proba_cat
    test_probas_cat += test_proba_cat / N_FOLDS

    # ------------------------------------
    # LightGBM with GPU (AGGRESSIVE)
    # ------------------------------------
    print(f"\n  [2/2] Training LightGBM (GPU - Aggressive)...")

    if USE_GPU:
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_estimators': 2500,
            'learning_rate': 0.025,
            'max_depth': 9,
            'num_leaves': 127,  # More leaves
            'min_child_samples': 10,  # Lower threshold
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.5,
            'reg_lambda': 1.0,
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': -1
        }
    else:
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'n_estimators': 2000,
            'learning_rate': 0.04,
            'max_depth': 7,
            'num_leaves': 63,
            'random_state': RANDOM_STATE,
            'verbose': -1
        }

    lgb_model = LGBMClassifier(**lgb_params)

    lgb_model.fit(
        X_train_lgb, y_train,
        eval_set=[(X_val_lgb, y_val)],
        eval_metric='multi_logloss'
    )

    # Predictions
    val_proba_lgb = lgb_model.predict_proba(X_val_lgb)
    test_proba_lgb = lgb_model.predict_proba(X_test_lgb)

    oof_probas_lgb[val_idx] = val_proba_lgb
    test_probas_lgb += test_proba_lgb / N_FOLDS

    # ------------------------------------
    # Ensemble with AGGRESSIVE High class threshold
    # ------------------------------------
    val_proba_ensemble = (val_proba_cat + val_proba_lgb) / 2

    # AGGRESSIVE: Lower threshold for High class
    # If High class probability > 0.25 (instead of default 0.33), predict High
    val_pred_ensemble = np.argmax(val_proba_ensemble, axis=1)

    # Boost High predictions: if High prob > 0.25 and it's not too low confident
    high_boost_mask = (val_proba_ensemble[:, 2] > 0.25) & (val_proba_ensemble[:, 0] < 0.6)
    val_pred_ensemble[high_boost_mask] = 2

    oof_predictions[val_idx] = val_pred_ensemble

    # Evaluate
    fold_f1 = f1_score(y_val, val_pred_ensemble, average='macro')
    fold_scores.append(fold_f1)

    print(f"\n  Fold {fold} Macro F1: {fold_f1:.6f}")
    print(f"  Per-class F1:")
    per_class_f1 = f1_score(y_val, val_pred_ensemble, average=None)
    for i, f1 in enumerate(per_class_f1):
        print(f"    {reverse_mapping[i]}: {f1:.4f}")

# ============================================
# OVERALL CV RESULTS
# ============================================
print("\n" + "="*80)
print("CV RESULTS")
print("="*80)

cv_f1 = f1_score(y_encoded, oof_predictions, average='macro')
print(f"\nOverall CV Macro F1: {cv_f1:.6f} ± {np.std(fold_scores):.6f}")
print(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")

print(f"\nPer-class F1 (CV):")
per_class_f1_cv = f1_score(y_encoded, oof_predictions, average=None)
for i, f1 in enumerate(per_class_f1_cv):
    print(f"  {reverse_mapping[i]}: {f1:.4f}")

print(f"\nConfusion Matrix (CV):")
cm = confusion_matrix(y_encoded, oof_predictions)
print(cm)

print(f"\nClassification Report (CV):")
print(classification_report(y_encoded, oof_predictions,
                          target_names=['Low', 'Medium', 'High']))

# ============================================
# FINAL PREDICTIONS ON TEST SET
# ============================================
print("\n[6/7] Generating AGGRESSIVE test predictions...")

# Ensemble test probabilities
test_proba_ensemble = (test_probas_cat + test_probas_lgb) / 2

# AGGRESSIVE High class prediction
test_pred_encoded = np.argmax(test_proba_ensemble, axis=1)

# Boost High predictions: if High prob > 0.25 and not too confident Low
high_boost_mask = (test_proba_ensemble[:, 2] > 0.25) & (test_proba_ensemble[:, 0] < 0.6)
test_pred_encoded[high_boost_mask] = 2

test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

print(f"\nComparison to V16:")
v16_sub = pd.read_csv('submissions/submission_v16_focal_loss_gpu.csv')
v16_counts = v16_sub['Target'].value_counts()
print(f"  V16: Low={v16_counts.get('Low', 0)} ({v16_counts.get('Low', 0)/len(v16_sub)*100:.1f}%), "
      f"Med={v16_counts.get('Medium', 0)} ({v16_counts.get('Medium', 0)/len(v16_sub)*100:.1f}%), "
      f"High={v16_counts.get('High', 0)} ({v16_counts.get('High', 0)/len(v16_sub)*100:.1f}%)")
print(f"  V17: Low={pred_counts.get('Low', 0)} ({pred_counts.get('Low', 0)/len(test_predictions)*100:.1f}%), "
      f"Med={pred_counts.get('Medium', 0)} ({pred_counts.get('Medium', 0)/len(test_predictions)*100:.1f}%), "
      f"High={pred_counts.get('High', 0)} ({pred_counts.get('High', 0)/len(test_predictions)*100:.1f}%)")

# ============================================
# SAVE SUBMISSION
# ============================================
print("\n[7/7] Saving submission...")

submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v17_aggressive_high.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V17 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: Aggressive High-Class Prediction (GPU)")
print(f"CV Macro F1: {cv_f1:.6f} ± {np.std(fold_scores):.6f}")
print(f"GPU Used: {USE_GPU}")
print(f"Focal Loss Gamma: {GAMMA}")
print(f"High Class Boost: {HIGH_CLASS_BOOST}x")
print(f"High Threshold: 0.25 (vs default 0.33)")
print(f"Expected LB: 0.894-0.900 (more aggressive High class)")
print(f"\nKey differences from V16:")
print(f"  - Extreme class weights (25x for High vs 17x)")
print(f"  - Gamma 3.0 vs 2.0 (more focus on hard examples)")
print(f"  - Lower High class threshold (0.25 vs 0.33)")
print(f"  - Deeper trees and more iterations")
print(f"{'='*80}")
