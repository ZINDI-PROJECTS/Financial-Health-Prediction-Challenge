"""
V12: 3-MODEL ENSEMBLE - CatBoost + LightGBM + XGBoost

Current ensemble: CatBoost + LightGBM (50/50)
Problem: Only 2 models, limited diversity

Solution: Add XGBoost
- Different boosting algorithm (dart/gbtree)
- Different regularization approach
- More diverse predictions → better ensemble

Expected: Model diversity reduces overfitting, better LB
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from scipy.optimize import minimize
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
print("V12: 3-MODEL ENSEMBLE - CatBoost + LightGBM + XGBoost")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/4] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# Encode categoricals for all models
X_encoded = X.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_encoded[col] = le.transform(X[col])
    X_test_encoded[col] = le.transform(X_test[col])

# ============================================
# CROSS-VALIDATION
# ============================================
print("\n[2/4] Training 3-model ensemble with 5-fold CV...")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1_scores = []
all_y_true = []
all_y_proba_blend = []

print(f"\n{'='*80}")
print("CROSS-VALIDATION: CatBoost + LightGBM + XGBoost")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_encoded, y_encoded), 1):
    print(f"\nFold {fold_idx}/5:")

    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # Train CatBoost
    print("  Training CatBoost...", end=' ')
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cat_proba = cat_model.predict_proba(X_val)
    print("Done")

    # Train LightGBM
    print("  Training LightGBM...", end=' ')
    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        metric='multi_logloss',
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1,
        early_stopping_rounds=50
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    lgb_proba = lgb_model.predict_proba(X_val)
    print("Done")

    # Train XGBoost
    print("  Training XGBoost...", end=' ')
    # Calculate sample weights from class weights
    sample_weights = np.array([class_weights[label] for label in y_train])

    xgb_model = XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        eval_metric='mlogloss',
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbosity=0,
        early_stopping_rounds=50
    )
    xgb_model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    xgb_proba = xgb_model.predict_proba(X_val)
    print("Done")

    # Blend 3 models (equal weight)
    blend_proba = (cat_proba + lgb_proba + xgb_proba) / 3

    # Optimize thresholds
    thresholds = optimize_thresholds(y_val, blend_proba)
    y_pred = predict_with_thresholds(blend_proba, thresholds)

    # Metrics
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    fold_f1_scores.append(f1)
    all_y_true.extend(y_val)
    all_y_proba_blend.append(blend_proba)

    print(f"  3-Model F1: {f1:.6f} | Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

# Aggregate
all_y_proba_blend = np.vstack(all_y_proba_blend)
all_y_true = np.array(all_y_true)

global_thresholds = optimize_thresholds(all_y_true, all_y_proba_blend)
y_pred_global = predict_with_thresholds(all_y_proba_blend, global_thresholds)

cv_f1 = f1_score(all_y_true, y_pred_global, average='macro')
f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

# Confusion matrix
cm = confusion_matrix(all_y_true, y_pred_global)

print(f"\n{'='*80}")
print("V12 CV RESULTS")
print(f"{'='*80}")
print(f"\nCV Macro-F1: {cv_f1:.6f}")
print(f"\nPer-class F1:")
print(f"  Low (0):    {f1_per_class[0]:.4f}")
print(f"  Medium (1): {f1_per_class[1]:.4f}")
print(f"  High (2):   {f1_per_class[2]:.4f}")
print(f"\nConfusion Matrix:")
print(f"         Pred_Low  Pred_Med  Pred_High")
print(f"True_Low    {cm[0,0]:5d}    {cm[0,1]:5d}     {cm[0,2]:5d}")
print(f"True_Med    {cm[1,0]:5d}    {cm[1,1]:5d}     {cm[1,2]:5d}")
print(f"True_High   {cm[2,0]:5d}    {cm[2,1]:5d}     {cm[2,2]:5d}")

# High class analysis
high_total = cm[2].sum()
high_correct = cm[2, 2]
high_to_med = cm[2, 1]

print(f"\nHigh Class Analysis:")
print(f"  Total: {high_total}")
print(f"  Correct: {high_correct} ({high_correct/high_total*100:.1f}%)")
print(f"  Confused with Medium: {high_to_med} ({high_to_med/high_total*100:.1f}%)")

# Compare
print(f"\n{'='*80}")
print("COMPARISON")
print(f"{'='*80}")
print(f"  V2 (Cat+LGB 50/50):          0.803 CV → 0.883 LB")
print(f"  V12 (Cat+LGB+XGB 33/33/33):  {cv_f1:.3f} CV → ???")
print(f"\nExpected: More model diversity → better generalization")

# ============================================
# TRAIN FINAL MODEL AND PREDICT
# ============================================
print(f"\n[3/4] Training final 3-model ensemble...")

# Train CatBoost
print("  Training CatBoost...", end=' ')
final_cat = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_cat.fit(X_encoded, y_encoded, verbose=False)
print("Done")

# Train LightGBM
print("  Training LightGBM...", end=' ')
final_lgb = LGBMClassifier(
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
final_lgb.fit(X_encoded, y_encoded)
print("Done")

# Train XGBoost
print("  Training XGBoost...", end=' ')
sample_weights = np.array([class_weights[label] for label in y_encoded])
final_xgb = XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    eval_metric='mlogloss',
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    random_state=RANDOM_STATE,
    verbosity=0
)
final_xgb.fit(X_encoded, y_encoded, sample_weight=sample_weights)
print("Done")

# Predict on test
print("\n[4/4] Generating predictions...")
cat_test_proba = final_cat.predict_proba(X_test_encoded)
lgb_test_proba = final_lgb.predict_proba(X_test_encoded)
xgb_test_proba = final_xgb.predict_proba(X_test_encoded)

blend_test_proba = (cat_test_proba + lgb_test_proba + xgb_test_proba) / 3

test_pred_encoded = predict_with_thresholds(blend_test_proba, global_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v12_xgboost_ensemble.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V12 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: 3-model ensemble (Cat+LGB+XGB)")
print(f"CV Macro-F1: {cv_f1:.6f}")
print(f"High→Medium confusion: {high_to_med/high_total*100:.1f}%")
print(f"\nExpected: More diversity → better LB performance")
print(f"{'='*80}")
