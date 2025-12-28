"""
V15: ADASYN - ADAPTIVE SYNTHETIC SAMPLING

V13 Failed (0.888): OOF optimization overfits to training distribution
V14 Risk: Also OOF-based, likely to fail

NEW APPROACH: ADASYN (Adaptive Synthetic Sampling)
- Smarter than SMOTE: Generates MORE samples where classification is harder
- Focuses on decision boundary (like Borderline-SMOTE but adaptive)
- Adapts density of synthetic samples based on difficulty
- Should generalize better than regular SMOTE

Strategy:
- Use ADASYN for High class oversampling
- More aggressive High class weight (8.0 instead of 7.0)
- Generate synthetic samples adaptively near misclassified regions

Expected: 0.895-0.900 LB (targets High class confusion directly)
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
from imblearn.over_sampling import ADASYN
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
print("V15: ADASYN - ADAPTIVE SYNTHETIC SAMPLING FOR HIGH CLASS")
print("="*80)
print(f"V10: 0.892 LB (50/50 blend)")
print(f"V13: 0.888 LB (OOF optimization FAILED)")
print(f"V15: ADASYN (adaptive, targets hard examples)")
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

print(f"\nOriginal class distribution:")
for cls, count in y.value_counts().items():
    pct = count / len(y) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# ============================================
# 2. ENCODE FOR ADASYN
# ============================================
print("\n[2/5] Encoding categorical features for ADASYN...")

X_encoded = X.copy()
X_test_encoded = X_test.copy()

for col in categorical_cols:
    le = LabelEncoder()
    combined = pd.concat([X[col], X_test[col]])
    le.fit(combined)
    X_encoded[col] = le.transform(X[col])
    X_test_encoded[col] = le.transform(X_test[col])

# ============================================
# 3. APPLY ADASYN
# ============================================
print("\n[3/5] Applying ADASYN to High class...")

current_high = (y_encoded == 2).sum()
target_high = 1000

sampling_strategy = {
    0: (y_encoded == 0).sum(),
    1: (y_encoded == 1).sum(),
    2: target_high
}

print(f"ADASYN sampling strategy:")
print(f"  Low:    {sampling_strategy[0]:,} (unchanged)")
print(f"  Medium: {sampling_strategy[1]:,} (unchanged)")
print(f"  High:   {sampling_strategy[2]:,} ({target_high - current_high:,} adaptive synthetic samples)")

# ADASYN: n_neighbors for finding difficult samples
adasyn = ADASYN(
    sampling_strategy=sampling_strategy,
    random_state=RANDOM_STATE,
    n_neighbors=5  # Find 5 nearest neighbors to determine difficulty
)

X_resampled, y_resampled = adasyn.fit_resample(X_encoded, y_encoded)
X_resampled = pd.DataFrame(X_resampled, columns=X_encoded.columns)

print(f"\nAfter ADASYN:")
print(f"  Total samples: {len(X_resampled):,} (was {len(X_encoded):,})")
for cls in [0, 1, 2]:
    count = (y_resampled == cls).sum()
    pct = count / len(y_resampled) * 100
    cls_name = reverse_mapping[cls]
    print(f"  {cls_name}: {count:,} ({pct:.1f}%)")

print(f"\nKey Difference from SMOTE:")
print(f"  - SMOTE: Uniform sampling across all High class samples")
print(f"  - ADASYN: MORE samples where classification is difficult")
print(f"  - Result: Better decision boundary, less overfitting")

# ============================================
# 4. CROSS-VALIDATION
# ============================================
print("\n[4/5] Training ensemble with 5-fold CV...")

# More aggressive High class weight
class_weights = {0: 1.0, 1: 2.5, 2: 8.0}  # Increased from 7.0
print(f"Class weights: {class_weights}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1_scores = []
all_y_true = []
all_y_proba_blend = []

print(f"\n{'='*80}")
print("CROSS-VALIDATION: CatBoost + LightGBM on ADASYN Data")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_resampled, y_resampled), 1):
    print(f"\nFold {fold_idx}/5:")

    X_train, X_val = X_resampled.iloc[train_idx], X_resampled.iloc[val_idx]
    y_train, y_val = y_resampled[train_idx], y_resampled[val_idx]

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

    # Blend
    blend_proba = (cat_proba + lgb_proba) / 2

    # Optimize thresholds
    thresholds = optimize_thresholds(y_val, blend_proba)
    y_pred = predict_with_thresholds(blend_proba, thresholds)

    # Metrics
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    fold_f1_scores.append(f1)
    all_y_true.extend(y_val)
    all_y_proba_blend.append(blend_proba)

    print(f"  Blend F1: {f1:.6f} | Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

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
print("V15 CV RESULTS (ON ADASYN DATA)")
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
print("COMPARISON TO PREVIOUS VERSIONS")
print(f"{'='*80}")
print(f"  V9 (SMOTE):       0.845 CV → 0.888 LB")
print(f"  V11 (Borderline): 0.847 CV → ???")
print(f"  V15 (ADASYN):     {cv_f1:.3f} CV → ???")
print(f"\nKey: ADASYN adapts to difficult samples, should generalize better")

# ============================================
# 5. TRAIN FINAL MODEL AND PREDICT
# ============================================
print(f"\n[5/5] Training final model on ADASYN data...")

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
final_cat.fit(X_resampled, y_resampled, verbose=False)
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
final_lgb.fit(X_resampled, y_resampled)
print("Done")

# Predict on test
cat_test_proba = final_cat.predict_proba(X_test_encoded)
lgb_test_proba = final_lgb.predict_proba(X_test_encoded)

blend_test_proba = (cat_test_proba + lgb_test_proba) / 2

test_pred_encoded = predict_with_thresholds(blend_test_proba, global_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V9 and V10
print(f"\nComparison:")
print(f"  V9 (SMOTE):  High=98")
print(f"  V10 (blend): High=94")
print(f"  V13 (optim): High=93 → 0.888 LB (FAILED)")
print(f"  V15 (ADASYN): High={pred_counts.get('High', 0)}")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v15_adasyn_high_class.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V15 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: ADASYN (adaptive synthetic sampling) + aggressive weights")
print(f"CV Macro-F1: {cv_f1:.6f}")
print(f"High→Medium confusion: {high_to_med/high_total*100:.1f}%")
print(f"\nExpected LB: 0.895-0.900")
print(f"Why: ADASYN adapts to hard examples, less overfitting than SMOTE")
print(f"{'='*80}")
