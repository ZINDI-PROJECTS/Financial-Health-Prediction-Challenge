"""
STEP 6: Model Blending
Blend CatBoost + LightGBM probabilities with threshold tuning
"""
import sys
sys.path.append('src')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from model import optimize_thresholds, predict_with_thresholds
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("STEP 6: MODEL BLENDING - CatBoost + LightGBM")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Prepare features
y = train_df['Target']
X_cat = train_df.drop(['Target', 'ID'], axis=1)
X_test_cat = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID'].values

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_encoded = y.map(label_mapping)

# Feature types
categorical_cols = X_cat.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_cat.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle missing values
for col in categorical_cols:
    X_cat[col] = X_cat[col].fillna('Missing')
    X_test_cat[col] = X_test_cat[col].fillna('Missing')

for col in numerical_cols:
    median_val = X_cat[col].median()
    X_cat[col] = X_cat[col].fillna(median_val)
    X_test_cat[col] = X_test_cat[col].fillna(median_val)

# For CatBoost
for col in categorical_cols:
    X_cat[col] = X_cat[col].astype(str)
    X_test_cat[col] = X_test_cat[col].astype(str)

# For LightGBM - encode categoricals
X_lgb = X_cat.copy()
X_test_lgb = X_test_cat.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X_cat[col], X_test_cat[col]])
    le.fit(combined)
    X_lgb[col] = le.transform(X_cat[col])
    X_test_lgb[col] = le.transform(X_test_cat[col])

print(f"Features shape: {X_cat.shape}")

# Best class weights
class_weights_dict = {0: 1.0, 1: 2.5, 2: 7.0}
sample_weights = y_encoded.map(class_weights_dict)

# Cross-validation with blending
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

print(f"\n{'='*80}")
print(f"BLENDING: Simple Mean of CatBoost + LightGBM Probabilities")
print(f"{'='*80}")

fold_f1_scores = []
all_y_true = []
all_y_proba_blend = []
all_y_proba_cat = []
all_y_proba_lgb = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cat, y_encoded), 1):
    print(f"\nFold {fold_idx}/{n_folds}...")

    # CatBoost
    X_train_cat, X_val_cat = X_cat.iloc[train_idx], X_cat.iloc[val_idx]
    X_train_lgb, X_val_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # Train CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights_dict,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(
        X_train_cat, y_train,
        cat_features=categorical_cols,
        eval_set=(X_val_cat, y_val),
        verbose=False
    )
    cat_proba = cat_model.predict_proba(X_val_cat)

    # Train LightGBM
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=class_weights_dict,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1,
        early_stopping_rounds=50
    )
    lgb_model.fit(
        X_train_lgb, y_train,
        eval_set=[(X_val_lgb, y_val)]
    )
    lgb_proba = lgb_model.predict_proba(X_val_lgb)

    # Blend probabilities (simple mean)
    blend_proba = (cat_proba + lgb_proba) / 2

    # Raw predictions
    y_pred_raw = np.argmax(blend_proba, axis=1)
    f1_raw = f1_score(y_val, y_pred_raw, average='macro')

    # Tune thresholds
    thresholds = optimize_thresholds(y_val.values, blend_proba)
    y_pred_tuned = predict_with_thresholds(blend_proba, thresholds)
    f1_tuned = f1_score(y_val, y_pred_tuned, average='macro')

    print(f"  CatBoost F1: {f1_score(y_val, np.argmax(cat_proba, axis=1), average='macro'):.4f}")
    print(f"  LightGBM F1: {f1_score(y_val, np.argmax(lgb_proba, axis=1), average='macro'):.4f}")
    print(f"  Blend F1:    {f1_tuned:.4f} (raw: {f1_raw:.4f})")

    fold_f1_scores.append(f1_tuned)
    all_y_true.extend(y_val.values)
    all_y_proba_blend.append(blend_proba)
    all_y_proba_cat.append(cat_proba)
    all_y_proba_lgb.append(lgb_proba)

# Aggregate
all_y_proba_blend = np.vstack(all_y_proba_blend)
all_y_proba_cat = np.vstack(all_y_proba_cat)
all_y_proba_lgb = np.vstack(all_y_proba_lgb)
all_y_true = np.array(all_y_true)

# Global threshold optimization
global_thresholds = optimize_thresholds(all_y_true, all_y_proba_blend)
y_pred_global = predict_with_thresholds(all_y_proba_blend, global_thresholds)

# Metrics
mean_f1 = np.mean(fold_f1_scores)
std_f1 = np.std(fold_f1_scores)
f1_global = f1_score(all_y_true, y_pred_global, average='macro')
f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

# Compare to individual models
cat_pred_global = predict_with_thresholds(all_y_proba_cat, optimize_thresholds(all_y_true, all_y_proba_cat))
lgb_pred_global = predict_with_thresholds(all_y_proba_lgb, optimize_thresholds(all_y_true, all_y_proba_lgb))
cat_f1 = f1_score(all_y_true, cat_pred_global, average='macro')
lgb_f1 = f1_score(all_y_true, lgb_pred_global, average='macro')

print(f"\n{'='*80}")
print("STEP 6 RESULTS: BLENDING PERFORMANCE")
print(f"{'='*80}\n")
print(f"{'Model':<20} {'Global Macro-F1':>15} {'F1-Low':>8} {'F1-Med':>8} {'F1-High':>8}")
print("-" * 80)
print(f"{'CatBoost (solo)':<20} {cat_f1:>15.6f} "
      f"{f1_score(all_y_true, cat_pred_global, average=None)[0]:>8.4f} "
      f"{f1_score(all_y_true, cat_pred_global, average=None)[1]:>8.4f} "
      f"{f1_score(all_y_true, cat_pred_global, average=None)[2]:>8.4f}")
print(f"{'LightGBM (solo)':<20} {lgb_f1:>15.6f} "
      f"{f1_score(all_y_true, lgb_pred_global, average=None)[0]:>8.4f} "
      f"{f1_score(all_y_true, lgb_pred_global, average=None)[1]:>8.4f} "
      f"{f1_score(all_y_true, lgb_pred_global, average=None)[2]:>8.4f}")
print(f"{'Blend (Cat+LGB)':<20} {f1_global:>15.6f} "
      f"{f1_per_class[0]:>8.4f} {f1_per_class[1]:>8.4f} {f1_per_class[2]:>8.4f}")

gain_vs_cat = f1_global - cat_f1
gain_vs_lgb = f1_global - lgb_f1

print(f"\n{'='*80}")
print(f"BLENDING GAIN:")
print(f"  vs CatBoost: {gain_vs_cat:+.6f}")
print(f"  vs LightGBM: {gain_vs_lgb:+.6f}")
print(f"\n  Final CV Macro-F1: {f1_global:.6f}")
print(f"  Per-class: Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")
print(f"{'='*80}")

if f1_global >= 0.90:
    print(f"\n✓ TARGET REACHED: CV Macro-F1 >= 0.90")
else:
    print(f"\n⚠ TARGET NOT REACHED: CV Macro-F1 = {f1_global:.6f} < 0.90")
    print(f"  Gap to close: {0.90 - f1_global:.6f}")
