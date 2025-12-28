"""
V7: OPTUNA CLASS WEIGHT OPTIMIZATION

Problem diagnosed:
- V2-V6 all stuck at ~0.80 CV
- High class confused with Medium: 132/470 samples (28%)
- Decision boundary too conservative

Hypothesis:
- Current weights (1.0, 2.5, 7.0) are suboptimal
- V6 country interactions ARE valuable but need right tuning
- Optimal weights might be (1, 2-5, 8-15) to be more aggressive on High

Strategy:
- Use Optuna TPE sampler to search class weight space
- Fix Low=1.0, search Medium=[1-5], High=[5-15]
- Objective: Maximize CV Macro-F1
- 100 trials to find global optimum
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier
from scipy.optimize import minimize
import optuna
from optuna.samplers import TPESampler
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
print("V7: OPTUNA CLASS WEIGHT OPTIMIZATION")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/3] Loading data with V6 features...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# 2. OPTUNA OBJECTIVE FUNCTION
# ============================================

def objective(trial):
    """
    Optuna objective: Find optimal class weights

    Search space:
    - Low weight: FIXED at 1.0
    - Medium weight: [1, 5] (integer)
    - High weight: [5, 15] (integer)

    Returns: Mean CV Macro-F1 (3-fold for speed)
    """
    # Suggest class weights
    weight_low = 1.0  # Fixed
    weight_medium = trial.suggest_int('weight_medium', 1, 5)
    weight_high = trial.suggest_int('weight_high', 5, 15)

    class_weights = {0: weight_low, 1: weight_medium, 2: weight_high}

    # 3-fold CV for speed (faster than 5-fold)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    fold_f1_scores = []

    for train_idx, val_idx in skf.split(X, y_encoded):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

        # Train CatBoost
        model = CatBoostClassifier(
            loss_function='MultiClass',
            eval_metric='TotalF1',
            class_weights=class_weights,
            iterations=800,  # Slightly reduced for speed
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

        # Macro F1
        f1 = f1_score(y_val, y_pred, average='macro')
        fold_f1_scores.append(f1)

    # Return mean F1
    mean_f1 = np.mean(fold_f1_scores)

    # Log trial
    trial.set_user_attr('mean_f1', mean_f1)
    trial.set_user_attr('std_f1', np.std(fold_f1_scores))

    return mean_f1


# ============================================
# 3. RUN OPTUNA STUDY
# ============================================
print("\n[2/3] Running Optuna study (100 trials)...")
print("Search space:")
print("  Low weight:    FIXED at 1.0")
print("  Medium weight: [1, 5] (integer)")
print("  High weight:   [5, 15] (integer)")
print("\nObjective: Maximize 3-fold CV Macro-F1")
print("Sampler: Tree-structured Parzen Estimator (TPE)")
print()

# Create study
study = optuna.create_study(
    direction='maximize',
    sampler=TPESampler(seed=RANDOM_STATE)
)

# Run optimization
study.optimize(objective, n_trials=100, show_progress_bar=True)

# Results
print(f"\n{'='*80}")
print("OPTUNA OPTIMIZATION RESULTS")
print(f"{'='*80}")

best_trial = study.best_trial
best_weights = {
    0: 1.0,
    1: best_trial.params['weight_medium'],
    2: best_trial.params['weight_high']
}

print(f"\nBest trial #{best_trial.number}:")
print(f"  Macro-F1: {best_trial.value:.6f}")
print(f"  Std:      {best_trial.user_attrs['std_f1']:.6f}")
print(f"\nOptimal class weights:")
print(f"  Low:    {best_weights[0]}")
print(f"  Medium: {best_weights[1]}")
print(f"  High:   {best_weights[2]}")
print(f"\nComparison to previous best:")
print(f"  V2 hardcoded (1, 2.5, 7):  0.803 CV")
print(f"  V7 optimized {tuple(best_weights.values())}: {best_trial.value:.3f} CV")
print(f"  Gain: {best_trial.value - 0.802901:+.6f}")

# Show top 5 trials
print(f"\n{'='*80}")
print("TOP 5 TRIALS")
print(f"{'='*80}")
print(f"{'Rank':<6} {'F1':<10} {'Medium':<8} {'High':<6} {'Trial':<6}")
print("-" * 80)

sorted_trials = sorted(study.trials, key=lambda t: t.value, reverse=True)
for i, trial in enumerate(sorted_trials[:5], 1):
    if trial.value is not None:
        print(f"{i:<6} {trial.value:<10.6f} {trial.params['weight_medium']:<8} "
              f"{trial.params['weight_high']:<6} #{trial.number}")

# ============================================
# 4. RETRAIN WITH OPTIMAL WEIGHTS (5-FOLD)
# ============================================
print(f"\n[3/3] Retraining with optimal weights (5-fold for final validation)...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_f1_scores = []
all_y_true = []
all_y_proba = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        class_weights=best_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )
    model.fit(X_train, y_train, cat_features=categorical_cols,
              eval_set=(X_val, y_val), verbose=False)

    y_proba = model.predict_proba(X_val)
    thresholds = optimize_thresholds(y_val.values, y_proba)
    y_pred = predict_with_thresholds(y_proba, thresholds)

    f1 = f1_score(y_val, y_pred, average='macro')
    f1_per_class = f1_score(y_val, y_pred, average=None)

    fold_f1_scores.append(f1)
    all_y_true.extend(y_val.values)
    all_y_proba.append(y_proba)

    print(f"Fold {fold_idx}: {f1:.6f} | Low={f1_per_class[0]:.4f}, Med={f1_per_class[1]:.4f}, High={f1_per_class[2]:.4f}")

# Aggregate
all_y_proba = np.vstack(all_y_proba)
all_y_true = np.array(all_y_true)

global_thresholds = optimize_thresholds(all_y_true, all_y_proba)
y_pred_global = predict_with_thresholds(all_y_proba, global_thresholds)

cv_f1 = f1_score(all_y_true, y_pred_global, average='macro')
f1_per_class = f1_score(all_y_true, y_pred_global, average=None)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(all_y_true, y_pred_global)

print(f"\n{'='*80}")
print("V7 FINAL CV RESULTS (5-FOLD)")
print(f"{'='*80}")
print(f"\nCV Macro-F1: {cv_f1:.6f} ± {np.std(fold_f1_scores):.6f}")
print(f"\nPer-class F1:")
print(f"  Low (0):    {f1_per_class[0]:.4f}")
print(f"  Medium (1): {f1_per_class[1]:.4f}")
print(f"  High (2):   {f1_per_class[2]:.4f}")
print(f"\nConfusion Matrix:")
print(f"         Pred_Low  Pred_Med  Pred_High")
print(f"True_Low    {cm[0,0]:5d}    {cm[0,1]:5d}     {cm[0,2]:5d}")
print(f"True_Med    {cm[1,0]:5d}    {cm[1,1]:5d}     {cm[1,2]:5d}")
print(f"True_High   {cm[2,0]:5d}    {cm[2,1]:5d}     {cm[2,2]:5d}")

# Check High class confusion improvement
high_total = cm[2].sum()
high_correct = cm[2, 2]
high_to_med = cm[2, 1]
high_to_low = cm[2, 0]

print(f"\nHigh Class Analysis:")
print(f"  Total High samples: {high_total}")
print(f"  Correctly predicted: {high_correct} ({high_correct/high_total*100:.1f}%)")
print(f"  Confused with Medium: {high_to_med} ({high_to_med/high_total*100:.1f}%) ← KEY METRIC")
print(f"  Confused with Low: {high_to_low} ({high_to_low/high_total*100:.1f}%)")

print(f"\n{'='*80}")
print("BREAKTHROUGH CHECK")
print(f"{'='*80}")

if cv_f1 > 0.810:
    print("🔥 BREAKTHROUGH! V7 significantly beats all previous versions!")
    print(f"Expected public LB: ~{cv_f1 + 0.08:.3f}")
    if cv_f1 + 0.08 >= 0.90:
        print("🎯 PROJECTED TO HIT 0.90+ TARGET!")
elif cv_f1 > 0.802901:
    print("✓ V7 beats V2! Optimal weights found.")
    print(f"Expected public LB: ~{cv_f1 + 0.08:.3f}")
else:
    print("⚠ Optuna didn't break the ceiling. Class weights not the bottleneck.")
    print("Next: Try AutoGluon or focal loss implementation.")

# Train final model
print(f"\nTraining final model on 100% of data...")
final_model = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='TotalF1',
    class_weights=best_weights,
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    random_state=RANDOM_STATE,
    verbose=False
)
final_model.fit(X, y_encoded, cat_features=categorical_cols, verbose=False)

# Predict
test_proba = final_model.predict_proba(X_test)
test_pred_encoded = predict_with_thresholds(test_proba, global_thresholds)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v7_optuna_weights.csv'
submission.to_csv(filename, index=False)

pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

print(f"\n{'='*80}")
print("V7 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Optimal weights: {best_weights}")
print(f"CV Macro-F1: {cv_f1:.6f}")
print(f"High→Medium confusion: {high_to_med/high_total*100:.1f}%")
print(f"{'='*80}")
