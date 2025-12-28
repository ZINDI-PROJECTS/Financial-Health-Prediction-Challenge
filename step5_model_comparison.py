"""
STEP 5: Model Comparison
Compare CatBoost vs LightGBM vs XGBoost
Select top 2 models for blending
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
from xgboost import XGBClassifier
from model import optimize_thresholds, predict_with_thresholds
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("STEP 5: MODEL COMPARISON")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

# Prepare features
y = train_df['Target']
X = train_df.drop(['Target', 'ID'], axis=1)
X_test = test_df.drop(['ID'], axis=1)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_encoded = y.map(label_mapping)

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle missing values
for col in categorical_cols:
    X[col] = X[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')

for col in numerical_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# For LightGBM and XGBoost: encode categoricals
print("\nEncoding categorical features for LightGBM/XGBoost...")
X_encoded = X.copy()
X_test_encoded = X_test.copy()

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Fit on combined train+test to handle unseen values
    combined = pd.concat([X[col].astype(str), X_test[col].astype(str)])
    le.fit(combined)
    X_encoded[col] = le.transform(X[col].astype(str))
    X_test_encoded[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

# For CatBoost: keep as string
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"Features shape: {X.shape}")
print(f"Categorical features: {len(categorical_cols)}")

# Best class weights from STEP 2
class_weights_dict = {0: 1.0, 1: 2.5, 2: 7.0}

# Define models
models = {
    'CatBoost': {
        'model': CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='TotalF1',
            class_weights=class_weights_dict,
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_state=RANDOM_STATE,
            verbose=False,
            early_stopping_rounds=50
        ),
        'X': X,
        'X_test': X_test,
        'cat_features': categorical_cols
    },
    'LightGBM': {
        'model': LGBMClassifier(
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
        ),
        'X': X_encoded,
        'X_test': X_test_encoded,
        'cat_features': None
    },
    'XGBoost': {
        'model': XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            eval_metric='mlogloss',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            random_state=RANDOM_STATE,
            verbosity=0,
            early_stopping_rounds=50
        ),
        'X': X_encoded,
        'X_test': X_test_encoded,
        'cat_features': None
    }
}

# Manual class weights for XGBoost (sample_weight approach)
class_counts = y_encoded.value_counts().sort_index()
sample_weights_xgb = y_encoded.map({0: 1.0, 1: 2.5, 2: 7.0})

# Cross-validation
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

results_summary = []

for model_name, model_config in models.items():
    print(f"\n{'='*80}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*80}")

    model_template = model_config['model']
    X_train = model_config['X']
    X_test_pred = model_config['X_test']
    cat_features = model_config['cat_features']

    fold_f1_scores = []
    all_y_true = []
    all_y_proba = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_encoded), 1):
        print(f"\nFold {fold_idx}/{n_folds}...", end=' ')

        X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_train_fold, y_val_fold = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        # Clone model
        if model_name == 'CatBoost':
            model = CatBoostClassifier(**model_template.get_params())
            model.fit(
                X_train_fold, y_train_fold,
                cat_features=cat_features,
                eval_set=(X_val_fold, y_val_fold),
                verbose=False
            )
        elif model_name == 'LightGBM':
            model = LGBMClassifier(**model_template.get_params())
            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)]
            )
        elif model_name == 'XGBoost':
            model = XGBClassifier(**model_template.get_params())
            sample_weights_fold = sample_weights_xgb.iloc[train_idx]
            model.fit(
                X_train_fold, y_train_fold,
                sample_weight=sample_weights_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False
            )

        # Predict probabilities
        y_proba = model.predict_proba(X_val_fold)

        # Raw predictions
        y_pred_raw = np.argmax(y_proba, axis=1)
        f1_raw = f1_score(y_val_fold, y_pred_raw, average='macro')

        # Tune thresholds
        thresholds = optimize_thresholds(y_val_fold.values, y_proba)
        y_pred_tuned = predict_with_thresholds(y_proba, thresholds)
        f1_tuned = f1_score(y_val_fold, y_pred_tuned, average='macro')

        fold_f1_scores.append(f1_tuned)
        all_y_true.extend(y_val_fold.values)
        all_y_proba.append(y_proba)

        print(f"F1={f1_tuned:.4f}")

    # Aggregate results
    all_y_proba = np.vstack(all_y_proba)
    all_y_true = np.array(all_y_true)

    # Global threshold optimization
    global_thresholds = optimize_thresholds(all_y_true, all_y_proba)
    y_pred_global = predict_with_thresholds(all_y_proba, global_thresholds)

    # Final metrics
    mean_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)
    f1_global = f1_score(all_y_true, y_pred_global, average='macro')
    f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

    print(f"\n{model_name} Results:")
    print(f"  Mean CV Macro-F1: {mean_f1:.6f} ± {std_f1:.6f}")
    print(f"  Global Macro-F1:  {f1_global:.6f}")
    print(f"  Per-class F1: Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

    results_summary.append({
        'model': model_name,
        'mean_f1': mean_f1,
        'std_f1': std_f1,
        'f1_global': f1_global,
        'f1_low': f1_per_class[0],
        'f1_medium': f1_per_class[1],
        'f1_high': f1_per_class[2]
    })

# Print comparison
print(f"\n{'='*80}")
print("STEP 5 SUMMARY: MODEL COMPARISON")
print(f"{'='*80}\n")
print(f"{'Model':<15} {'Mean CV F1':>12} {'Global F1':>10} {'F1-Low':>8} {'F1-Med':>8} {'F1-High':>8}")
print("-" * 80)

for r in sorted(results_summary, key=lambda x: x['f1_global'], reverse=True):
    print(f"{r['model']:<15} {r['mean_f1']:>12.6f} {r['f1_global']:>10.6f} "
          f"{r['f1_low']:>8.4f} {r['f1_medium']:>8.4f} {r['f1_high']:>8.4f}")

# Select top 2
top_2 = sorted(results_summary, key=lambda x: x['f1_global'], reverse=True)[:2]
print(f"\n{'='*80}")
print(f"TOP 2 MODELS FOR BLENDING:")
print(f"  1. {top_2[0]['model']}: {top_2[0]['f1_global']:.6f}")
print(f"  2. {top_2[1]['model']}: {top_2[1]['f1_global']:.6f}")
print(f"{'='*80}")
