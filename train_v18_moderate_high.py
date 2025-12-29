"""
V18: MODERATE HIGH-CLASS BOOST (GPU)

Strategy:
- Sweet spot between V16 (conservative) and V17 (too aggressive)
- Moderate class weights: 20x for High (vs V16: 17x, V17: 25x)
- Gamma=2.5 (between V16: 2.0 and V17: 3.0)
- Slightly lower threshold: 0.30 for High (vs default 0.33, V17: 0.25)
- Target: 7-8% High predictions (vs V16: 5.4%, V17: 13.5%)

Expected: Better High recall than V16, better precision than V17
Target LB: 0.895-0.900
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

# MODERATE Focal Loss parameters
GAMMA = 2.5  # Between V16 (2.0) and V17 (3.0)
HIGH_CLASS_BOOST = 20.0  # Between V16 (17x) and V17 (25x)
HIGH_THRESHOLD = 0.30  # Between default (0.33) and V17 (0.25)

print("="*80)
print("V18: MODERATE HIGH-CLASS BOOST (GPU)")
print("="*80)
print(f"Parameters: Gamma={GAMMA}, High Boost={HIGH_CLASS_BOOST}x, Threshold={HIGH_THRESHOLD}")

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

# MODERATE class weights
class_counts = np.bincount(y_encoded)
total = len(y_encoded)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
class_weights[2] = class_weights[2] * HIGH_CLASS_BOOST  # Moderate boost
print(f"Class weights (MODERATE): {class_weights}")

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
    # CatBoost with GPU (MODERATE)
    # ------------------------------------
    print(f"\n  [1/2] Training CatBoost (GPU - Moderate)...")

    if USE_GPU:
        cat_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'task_type': 'GPU',
            'devices': '0',
            'gpu_ram_part': 0.7,
            'class_weights': class_weights,
            'iterations': 2200,  # Between V16 (2000) and V17 (2500)
            'learning_rate': 0.03,
            'depth': 8,  # Between V16 (8) and V17 (9)
            'l2_leaf_reg': 4,  # Between V16 (5) and V17 (3)
            'random_state': RANDOM_STATE,
            'verbose': 100,
            'early_stopping_rounds': 120
        }
    else:
        cat_params = {
            'loss_function': 'MultiClass',
            'eval_metric': 'TotalF1:average=Macro',
            'class_weights': class_weights,
            'iterations': 1800,
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
    # LightGBM with GPU (MODERATE)
    # ------------------------------------
    print(f"\n  [2/2] Training LightGBM (GPU - Moderate)...")

    if USE_GPU:
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0,
            'n_estimators': 2200,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 95,  # Between V16 (63) and V17 (127)
            'min_child_samples': 15,  # Between V16 (20) and V17 (10)
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 0.75,  # Between V16 (1.0) and V17 (0.5)
            'reg_lambda': 1.5,  # Between V16 (2.0) and V17 (1.0)
            'random_state': RANDOM_STATE,
            'verbose': -1,
            'n_jobs': -1
        }
    else:
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'n_estimators': 1800,
            'learning_rate': 0.04,
            'max_depth': 7,
            'num_leaves': 47,
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
    # Ensemble with MODERATE High class threshold
    # ------------------------------------
    val_proba_ensemble = (val_proba_cat + val_proba_lgb) / 2

    # MODERATE: Threshold of 0.30 for High class
    val_pred_ensemble = np.argmax(val_proba_ensemble, axis=1)

    # Moderate boost: if High prob > 0.30 and not too confident Low
    high_boost_mask = (val_proba_ensemble[:, 2] > HIGH_THRESHOLD) & (val_proba_ensemble[:, 0] < 0.55)
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
print("\n[6/7] Generating MODERATE test predictions...")

# Ensemble test probabilities
test_proba_ensemble = (test_probas_cat + test_probas_lgb) / 2

# MODERATE High class prediction
test_pred_encoded = np.argmax(test_proba_ensemble, axis=1)

# Moderate boost: if High prob > 0.30 and not too confident Low
high_boost_mask = (test_proba_ensemble[:, 2] > HIGH_THRESHOLD) & (test_proba_ensemble[:, 0] < 0.55)
test_pred_encoded[high_boost_mask] = 2

test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# ============================================
# SAVE SUBMISSION
# ============================================
print("\n[7/7] Saving submission...")

submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v18_moderate_high.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V18 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: Moderate High-Class Boost (GPU)")
print(f"CV Macro F1: {cv_f1:.6f} ± {np.std(fold_scores):.6f}")
print(f"GPU Used: {USE_GPU}")
print(f"Focal Loss Gamma: {GAMMA}")
print(f"High Class Boost: {HIGH_CLASS_BOOST}x")
print(f"High Threshold: {HIGH_THRESHOLD}")
print(f"Expected LB: 0.895-0.900")
print(f"{'='*80}")
