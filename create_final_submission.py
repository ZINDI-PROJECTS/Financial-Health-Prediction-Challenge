"""
FINAL SUBMISSION GENERATOR
Model: CatBoost + LightGBM Blend with Threshold Tuning
CV Macro-F1: 0.803
Class Weights: Low=1.0, Medium=2.5, High=7.0
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# Fixed seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("FINAL SUBMISSION GENERATION")
print("Model: CatBoost + LightGBM Blend")
print("="*80)


def optimize_thresholds(y_true, y_proba, n_classes=3):
    """Find optimal decision thresholds to maximize macro-F1"""
    def objective(thresholds):
        predictions = np.argmax(y_proba - thresholds, axis=1)
        return -f1_score(y_true, predictions, average='macro')

    init_thresholds = np.zeros(n_classes)
    result = minimize(objective, init_thresholds, method='Nelder-Mead', options={'maxiter': 500})
    return result.x


def predict_with_thresholds(y_proba, thresholds):
    """Apply learned thresholds to probability predictions"""
    return np.argmax(y_proba - thresholds, axis=1)


# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')
sample_sub = pd.read_csv('data/raw/SampleSubmission.csv')

print(f"  Train: {train_df.shape}")
print(f"  Test:  {test_df.shape}")

# ============================================
# 2. PREPARE FEATURES
# ============================================
print("\n[2/6] Preparing features...")

y = train_df['Target']
X_cat = train_df.drop(['Target', 'ID'], axis=1)
X_test_cat = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID'].values

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
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

print(f"  Features: {X_cat.shape[1]}")
print(f"  Categorical: {len(categorical_cols)}")
print(f"  Numerical: {len(numerical_cols)}")

# ============================================
# 3. LEARN THRESHOLDS VIA CV
# ============================================
print("\n[3/6] Learning optimal thresholds via 5-fold CV...")

class_weights_dict = {0: 1.0, 1: 2.5, 2: 7.0}
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

all_y_true = []
all_y_proba_blend = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cat, y_encoded), 1):
    print(f"  Fold {fold_idx}/{n_folds}...", end=' ')

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
    cat_model.fit(X_train_cat, y_train, cat_features=categorical_cols,
                  eval_set=(X_val_cat, y_val), verbose=False)
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
    lgb_model.fit(X_train_lgb, y_train, eval_set=[(X_val_lgb, y_val)])
    lgb_proba = lgb_model.predict_proba(X_val_lgb)

    # Blend
    blend_proba = (cat_proba + lgb_proba) / 2

    all_y_true.extend(y_val.values)
    all_y_proba_blend.append(blend_proba)

    print("Done")

# Aggregate and learn global thresholds
all_y_proba_blend = np.vstack(all_y_proba_blend)
all_y_true = np.array(all_y_true)

global_thresholds = optimize_thresholds(all_y_true, all_y_proba_blend)
y_pred_cv = predict_with_thresholds(all_y_proba_blend, global_thresholds)
cv_f1 = f1_score(all_y_true, y_pred_cv, average='macro')

print(f"\n  Global thresholds: {global_thresholds}")
print(f"  CV Macro-F1: {cv_f1:.6f}")

# ============================================
# 4. TRAIN FINAL MODELS ON FULL DATA
# ============================================
print("\n[4/6] Training final models on 100% of training data...")

# CatBoost
print("  Training CatBoost...", end=' ')
final_cat_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights_dict,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_cat_model.fit(X_cat, y_encoded, cat_features=categorical_cols, verbose=False)
print("Done")

# LightGBM
print("  Training LightGBM...", end=' ')
final_lgb_model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    metric='multi_logloss',
    class_weight=class_weights_dict,
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbose=-1
)
final_lgb_model.fit(X_lgb, y_encoded)
print("Done")

# ============================================
# 5. GENERATE TEST PREDICTIONS
# ============================================
print("\n[5/6] Generating test predictions...")

# Predict probabilities
cat_test_proba = final_cat_model.predict_proba(X_test_cat)
lgb_test_proba = final_lgb_model.predict_proba(X_test_lgb)

# Blend
blend_test_proba = (cat_test_proba + lgb_test_proba) / 2

# Apply learned thresholds
test_predictions_encoded = predict_with_thresholds(blend_test_proba, global_thresholds)
test_predictions = [reverse_mapping[pred] for pred in test_predictions_encoded]

print(f"  Generated {len(test_predictions)} predictions")

# Print prediction distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\n  Predicted distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = (count / len(test_predictions)) * 100
    print(f"    {cls:8s}: {count:5d} ({pct:5.2f}%)")

# ============================================
# 6. CREATE SUBMISSION FILE
# ============================================
print("\n[6/6] Creating submission file...")

submission = pd.DataFrame({
    'ID': test_ids,
    'Target': test_predictions
})

# Validate
print(f"\n  Validation:")
print(f"    ✓ Shape: {submission.shape} (expected: {sample_sub.shape})")
print(f"    ✓ Columns: {list(submission.columns)} == {list(sample_sub.columns)}")
print(f"    ✓ IDs match: {all(submission['ID'] == test_ids)}")
print(f"    ✓ Valid labels: {set(test_predictions).issubset({'Low', 'Medium', 'High'})}")
print(f"    ✓ No nulls: {submission.isnull().sum().sum() == 0}")

# Save
filename = 'submission_v2_catboost_lgb_blend.csv'
filepath = f'submissions/{filename}'
submission.to_csv(filepath, index=False)

print(f"\n  Saved: {filepath}")

# ============================================
# FINAL REPORT
# ============================================
print("\n" + "="*80)
print("SUBMISSION COMPLETE")
print("="*80)
print(f"\nModel Configuration:")
print(f"  Algorithm:      CatBoost + LightGBM (simple mean blend)")
print(f"  Class Weights:  Low=1.0, Medium=2.5, High=7.0")
print(f"  CV Macro-F1:    {cv_f1:.6f}")
print(f"  Thresholds:     {global_thresholds}")
print(f"\nSubmission File:")
print(f"  Path:           {filepath}")
print(f"  Shape:          {submission.shape}")
print(f"  Predictions:    {len(test_predictions)}")
print(f"\nPredicted Class Distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = (count / len(test_predictions)) * 100
    print(f"  {cls:8s}: {count:5d} ({pct:5.2f}%)")
print("\n" + "="*80)
print("Ready for Zindi upload!")
print("="*80)
