"""
FINAL TRAINING - V3 with Engineered Features
Target: 0.895+ public LB (rank ≤50)
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
from engineer_features import prepare_engineered_data

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def optimize_thresholds(y_true, y_proba, n_classes=3):
    """Find optimal thresholds for macro-F1"""
    def objective(thresholds):
        predictions = np.argmax(y_proba - thresholds, axis=1)
        return -f1_score(y_true, predictions, average='macro')
    init_thresholds = np.zeros(n_classes)
    result = minimize(objective, init_thresholds, method='Nelder-Mead', options={'maxiter': 500})
    return result.x


def predict_with_thresholds(y_proba, thresholds):
    """Apply thresholds"""
    return np.argmax(y_proba - thresholds, axis=1)


print("="*80)
print("V3: ENGINEERED FEATURES + AGGRESSIVE HIGH-CLASS WEIGHTING")
print("="*80)

# Load and engineer features
print("\n[1/5] Loading data and engineering features...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_engineered_data(train_df, test_df)

# Encode target
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"\nFeatures: {X.shape[1]} (37 original + 7 engineered)")
print(f"Samples: {len(X):,}")

# Class distribution
class_dist = y.value_counts()
print(f"\nClass distribution:")
for cls in ['Low', 'Medium', 'High']:
    print(f"  {cls}: {class_dist[cls]:,} ({class_dist[cls]/len(y)*100:.1f}%)")

# Test different class weight configurations
print("\n[2/5] Testing class weight configurations...")

weight_configs = [
    {'name': 'Previous best (1, 2.5, 7)', 'weights': {0: 1.0, 1: 2.5, 2: 7.0}},
    {'name': 'Push High harder (1, 2, 10)', 'weights': {0: 1.0, 1: 2.0, 2: 10.0}},
    {'name': 'Extreme High focus (1, 2, 12)', 'weights': {0: 1.0, 1: 2.0, 2: 12.0}},
]

best_config = None
best_f1 = 0

for config in weight_configs:
    print(f"\n{'='*80}")
    print(f"Testing: {config['name']}")
    print(f"{'='*80}")

    class_weights = config['weights']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    fold_f1_scores = []
    all_y_true = []
    all_y_proba = []

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

        f1 = f1_score(y_val, y_pred, average='macro')
        fold_f1_scores.append(f1)

        all_y_true.extend(y_val.values)
        all_y_proba.append(y_proba)

        print(f"  Fold {fold_idx}: {f1:.6f}")

    # Aggregate
    all_y_proba = np.vstack(all_y_proba)
    all_y_true = np.array(all_y_true)

    global_thresholds = optimize_thresholds(all_y_true, all_y_proba)
    y_pred_global = predict_with_thresholds(all_y_proba, global_thresholds)

    cv_f1 = f1_score(all_y_true, y_pred_global, average='macro')
    f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

    print(f"\n  CV Macro-F1: {cv_f1:.6f}")
    print(f"  Per-class: Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

    if cv_f1 > best_f1:
        best_f1 = cv_f1
        best_config = config
        best_thresholds = global_thresholds
        best_f1_per_class = f1_per_class

print(f"\n{'='*80}")
print(f"BEST CONFIGURATION")
print(f"{'='*80}")
print(f"  Config: {best_config['name']}")
print(f"  Weights: {best_config['weights']}")
print(f"  CV Macro-F1: {best_f1:.6f}")
print(f"  Per-class: Low={best_f1_per_class[0]:.4f}, Med={best_f1_per_class[1]:.4f}, High={best_f1_per_class[2]:.4f}")
print(f"  Thresholds: {best_thresholds}")

# Compare to baseline
baseline_cv = 0.802901
gain = best_f1 - baseline_cv
print(f"\n  Gain vs baseline (0.803): {gain:+.6f}")

if best_f1 < baseline_cv:
    print("\n  ⚠ WARNING: Engineered features DECREASED performance!")
    print("  This suggests feature leakage or overfitting. Review feature engineering.")
elif gain < 0.005:
    print("\n  ⚠ Minimal gain. Features help but not dramatically.")
else:
    print(f"\n  ✓ Significant gain! Features are working.")

# Train final model on all data
print(f"\n[3/5] Training final model on 100% of data...")
final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=best_config['weights'],
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_model.fit(X, y_encoded, cat_features=categorical_cols, verbose=100)

# Feature importance
print(f"\n[4/5] Top 10 most important features:")
feature_importance = final_model.get_feature_importance()
feature_names = X.columns
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

for idx, row in importance_df.head(10).iterrows():
    marker = "🆕" if row['feature'] in ['profit_margin', 'expense_burden', 'revenue_per_age',
                                         'credit_access_score', 'formalization_score',
                                         'stability_index', 'income_dependency'] else "  "
    print(f"  {marker} {row['feature']:<35s}: {row['importance']:>8.2f}")

# Generate predictions
print(f"\n[5/5] Generating test predictions...")
test_proba = final_model.predict_proba(X_test)
test_pred_encoded = predict_with_thresholds(test_proba, best_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Prediction distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\n  Test predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"    {cls}: {count:,} ({pct:.1f}%)")

# Save submission
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v3_engineered_features.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("SUBMISSION V3 COMPLETE")
print(f"{'='*80}")
print(f"\nFile: {filename}")
print(f"CV Macro-F1: {best_f1:.6f}")
print(f"Class weights: {best_config['weights']}")
print(f"Gain vs baseline: {gain:+.6f}")
print(f"\nPer-class F1:")
print(f"  Low:    {best_f1_per_class[0]:.4f}")
print(f"  Medium: {best_f1_per_class[1]:.4f}")
print(f"  High:   {best_f1_per_class[2]:.4f}")
print(f"\nExpected Public LB: ~{best_f1 + 0.08:.3f} (if CV→LB gap holds)")
print(f"{'='*80}")
