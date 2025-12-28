"""
V4 TRAINING - REFINED FEATURES (Clean Signal Only)

Changes from V3:
- Removed 4 noisy features
- Kept only 3 clean, proven features
- Refined profit_margin implementation (log-scale + better capping)

Target: Beat V2 (0.883 public) by focusing on clean signal
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from engineer_features_v4 import prepare_data_v4

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
print("V4: REFINED FEATURES - CLEAN SIGNAL ONLY")
print("="*80)

# Load data
print("\n[1/4] Loading data and engineering V4 features...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v4(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"\nFeatures: {X.shape[1]} (40 total = 37 original + 3 refined)")
print(f"Samples: {len(X):,}")

# Class distribution
class_dist = y.value_counts()
print(f"\nClass distribution:")
for cls in ['Low', 'Medium', 'High']:
    print(f"  {cls}: {class_dist[cls]:,} ({class_dist[cls]/len(y)*100:.1f}%)")

# CV with best class weights from all experiments
print("\n[2/4] Training with 5-fold CV...")
print("Class weights: {0: 1.0, 1: 2.5, 2: 7.0} (proven best)")

class_weights = {0: 1.0, 1: 2.5, 2: 7.0}
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

fold_f1_scores = []
all_y_true = []
all_y_proba = []

print(f"\n{'='*80}")
print("CROSS-VALIDATION")
print(f"{'='*80}")

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # Train CatBoost
    model = CatBoostClassifier(
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
    model.fit(X_train, y_train, cat_features=categorical_cols,
              eval_set=(X_val, y_val), verbose=False)

    # Predict
    y_proba = model.predict_proba(X_val)
    thresholds = optimize_thresholds(y_val.values, y_proba)
    y_pred = predict_with_thresholds(y_proba, thresholds)

    # Metrics
    f1 = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    fold_f1_scores.append(f1)
    all_y_true.extend(y_val.values)
    all_y_proba.append(y_proba)

    print(f"Fold {fold_idx}: Macro-F1={f1:.6f} | "
          f"Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

# Aggregate
all_y_proba = np.vstack(all_y_proba)
all_y_true = np.array(all_y_true)

global_thresholds = optimize_thresholds(all_y_true, all_y_proba)
y_pred_global = predict_with_thresholds(all_y_proba, global_thresholds)

cv_f1 = f1_score(all_y_true, y_pred_global, average='macro')
f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

# Confusion matrix
cm = confusion_matrix(all_y_true, y_pred_global)

print(f"\n{'='*80}")
print("CV RESULTS")
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

# Compare to baselines
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")
baseline_v2 = 0.802901
baseline_v3 = 0.799367

print(f"  V2 (Blend):      0.803 CV → 0.883 Public")
print(f"  V3 (7 features): 0.799 CV → 0.872 Public")
print(f"  V4 (3 refined):  {cv_f1:.3f} CV")

gain_v2 = cv_f1 - baseline_v2
gain_v3 = cv_f1 - baseline_v3

print(f"\n  Gain vs V2:  {gain_v2:+.6f}")
print(f"  Gain vs V3:  {gain_v3:+.6f}")

if cv_f1 > baseline_v2:
    print(f"\n  ✓ V4 BEATS V2! Clean features are working.")
    print(f"  Expected public: ~{cv_f1 + 0.08:.3f} (if gap holds)")
elif cv_f1 > baseline_v3:
    print(f"\n  ✓ V4 beats V3. Simplification helped, but still below V2.")
    print(f"  Expected public: ~{cv_f1 + 0.08:.3f}")
else:
    print(f"\n  ⚠ V4 below both baselines. Features not helping.")

# Train final model
print(f"\n[3/4] Training final model on 100% of data...")
final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=class_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_model.fit(X, y_encoded, cat_features=categorical_cols, verbose=100)

# Feature importance
print(f"\n[4/4] Feature Importance (Top 15):")
importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': importance
}).sort_values('importance', ascending=False)

new_features = ['credit_access_score', 'profit_margin', 'business_maturity']
for idx, row in importance_df.head(15).iterrows():
    marker = "🆕" if row['feature'] in new_features else "  "
    print(f"  {marker} {row['feature']:<40s}: {row['importance']:>6.2f}")

# Check if new features are being used
new_feat_ranks = []
for feat in new_features:
    rank = (importance_df['feature'] == feat).idxmax()
    actual_rank = importance_df.index.get_loc(rank) + 1
    new_feat_ranks.append(actual_rank)
    print(f"\n  🆕 {feat}: Rank #{actual_rank}")

# Generate predictions
print(f"\nGenerating test predictions...")
test_proba = final_model.predict_proba(X_test)
test_pred_encoded = predict_with_thresholds(test_proba, global_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Prediction distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v4_refined_clean.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V4 SUBMISSION COMPLETE")
print(f"{'='*80}")
print(f"\nFile: {filename}")
print(f"CV Macro-F1: {cv_f1:.6f}")
print(f"Features: 3 refined (credit_access, profit_margin_log, business_maturity)")
print(f"Class weights: {class_weights}")
print(f"\nPer-class F1:")
print(f"  Low:    {f1_per_class[0]:.4f}")
print(f"  Medium: {f1_per_class[1]:.4f}")
print(f"  High:   {f1_per_class[2]:.4f}")
print(f"\n{'='*80}")
