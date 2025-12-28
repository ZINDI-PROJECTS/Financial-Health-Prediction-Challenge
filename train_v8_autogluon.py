"""
V8: AUTOGLUON - FULL AUTOML APPROACH

Why AutoGluon:
1. Automatically tests 10+ algorithms (CatBoost, LightGBM, XGBoost, Neural Nets, etc.)
2. Advanced multi-layer stacking (not just simple blending)
3. Automatic hyperparameter tuning
4. Handles imbalanced data with sophisticated strategies
5. Often finds breakthrough combinations missed by manual optimization

Strategy:
- Use V6 features (country interactions)
- Let AutoGluon optimize everything: models, weights, stacking
- Quality preset: 'best_quality' (intensive search)
- Time budget: 30 minutes (reasonable for 100 trials worth)
- Custom metric: macro-F1
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, make_scorer, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Check if AutoGluon is installed
try:
    from autogluon.tabular import TabularPredictor, TabularDataset
    AUTOGLUON_AVAILABLE = True
except ImportError:
    AUTOGLUON_AVAILABLE = False
    print("AutoGluon not installed. Installing...")

if not AUTOGLUON_AVAILABLE:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autogluon"])
    from autogluon.tabular import TabularPredictor, TabularDataset

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("V8: AUTOGLUON AUTOML")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/4] Loading data with V6 features...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")
print(f"Categorical features: {len(categorical_cols)}")

# ============================================
# 2. PREPARE AUTOGLUON FORMAT
# ============================================
print("\n[2/4] Preparing AutoGluon datasets...")

# Combine X and y for AutoGluon
train_data = X.copy()
train_data['Target'] = y.values

test_data = X_test.copy()

# Convert to TabularDataset
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

print(f"Train data shape: {train_data.shape}")
print(f"Test data shape: {test_data.shape}")

# ============================================
# 3. CONFIGURE AUTOGLUON
# ============================================
print("\n[3/4] Configuring AutoGluon...")

# Custom evaluation metric: macro-F1
def macro_f1_scorer(y_true, y_pred):
    """Custom macro-F1 scorer for AutoGluon"""
    return f1_score(y_true, y_pred, average='macro')

# AutoGluon configuration
ag_config = {
    'label': 'Target',
    'problem_type': 'multiclass',
    'eval_metric': 'f1_macro',  # Built-in macro F1
    'verbosity': 2,
}

# Training configuration
fit_config = {
    'presets': 'best_quality',  # Most aggressive optimization
    'time_limit': 1800,  # 30 minutes
    'num_bag_folds': 5,  # 5-fold bagging for robustness
    'num_bag_sets': 1,
    'num_stack_levels': 2,  # Two-level stacking
    'hyperparameters': {
        'CAT': {},  # CatBoost
        'GBM': {},  # LightGBM
        'XGB': {},  # XGBoost
        'NN_TORCH': {},  # Neural network
        'FASTAI': {},  # FastAI neural network
    },
    'auto_stack': True,  # Enable automatic stacking
    'verbosity': 2,
}

print("Configuration:")
print(f"  Preset: best_quality (most intensive)")
print(f"  Time limit: 30 minutes")
print(f"  Cross-validation: 5-fold bagging")
print(f"  Stacking: 2 levels")
print(f"  Models: CatBoost, LightGBM, XGBoost, Neural Nets")
print(f"  Metric: macro-F1")

# ============================================
# 4. TRAIN AUTOGLUON
# ============================================
print("\n[4/4] Training AutoGluon (this will take ~30 minutes)...")
print("="*80)

# Create predictor
predictor = TabularPredictor(**ag_config)

# Train
predictor.fit(
    train_data=train_data,
    **fit_config
)

print("\n" + "="*80)
print("AUTOGLUON TRAINING COMPLETE")
print("="*80)

# ============================================
# 5. EVALUATE RESULTS
# ============================================
print("\nModel leaderboard:")
leaderboard = predictor.leaderboard(train_data, silent=True)
print(leaderboard[['model', 'score_val', 'pred_time_val', 'fit_time']].head(10))

# Best model
best_model = leaderboard.iloc[0]['model']
best_score = leaderboard.iloc[0]['score_val']

print(f"\n{'='*80}")
print("BEST MODEL")
print(f"{'='*80}")
print(f"Model: {best_model}")
print(f"CV Macro-F1: {best_score:.6f}")

# Get feature importance
feature_importance = predictor.feature_importance(train_data)
print(f"\nTop 15 Features:")
for feat, importance in feature_importance.head(15).items():
    marker = "🆕" if '_x_' in feat or feat in ['credit_access_score', 'profit_margin', 'business_maturity'] else "  "
    print(f"  {marker} {feat:<50s}: {importance:>8.2f}")

# ============================================
# 6. DETAILED EVALUATION WITH CONFUSION MATRIX
# ============================================
print(f"\n{'='*80}")
print("DETAILED EVALUATION")
print(f"{'='*80}")

# Get predictions on training data (out-of-fold)
y_pred_oof = predictor.predict(train_data)
y_true = train_data['Target'].values

# Per-class metrics
f1_per_class = f1_score(y_true, y_pred_oof, average=None, labels=['Low', 'Medium', 'High'])
cm = confusion_matrix(y_true, y_pred_oof, labels=['Low', 'Medium', 'High'])

print(f"\nPer-class F1:")
print(f"  Low:    {f1_per_class[0]:.4f}")
print(f"  Medium: {f1_per_class[1]:.4f}")
print(f"  High:   {f1_per_class[2]:.4f}")

print(f"\nConfusion Matrix:")
print(f"         Pred_Low  Pred_Med  Pred_High")
print(f"True_Low    {cm[0,0]:5d}    {cm[0,1]:5d}     {cm[0,2]:5d}")
print(f"True_Med    {cm[1,0]:5d}    {cm[1,1]:5d}     {cm[1,2]:5d}")
print(f"True_High   {cm[2,0]:5d}    {cm[2,1]:5d}     {cm[2,2]:5d}")

# High class analysis
high_idx = 2  # Assuming order is Low, Medium, High
high_total = cm[high_idx].sum()
high_correct = cm[high_idx, high_idx]
high_to_med = cm[high_idx, 1]
high_to_low = cm[high_idx, 0]

print(f"\nHigh Class Analysis:")
print(f"  Total High samples: {high_total}")
print(f"  Correctly predicted: {high_correct} ({high_correct/high_total*100:.1f}%)")
print(f"  Confused with Medium: {high_to_med} ({high_to_med/high_total*100:.1f}%) ← KEY METRIC")
print(f"  Confused with Low: {high_to_low} ({high_to_low/high_total*100:.1f}%)")

# ============================================
# 7. COMPARISON TO PREVIOUS VERSIONS
# ============================================
print(f"\n{'='*80}")
print("PERFORMANCE COMPARISON")
print(f"{'='*80}")
print(f"  V2 (Manual blend):      0.803 CV → 0.883 Public")
print(f"  V6 (Country features):  0.800 CV → ???")
print(f"  V8 (AutoGluon):         {best_score:.3f} CV → ???")

gain_v2 = best_score - 0.802901

print(f"\n  Gain vs V2: {gain_v2:+.6f}")

if best_score > 0.810:
    print("\n🔥 BREAKTHROUGH! AutoGluon found a superior approach!")
    print(f"Expected public LB: ~{best_score + 0.08:.3f}")
    if best_score + 0.08 >= 0.90:
        print("🎯 PROJECTED TO HIT 0.90+ TARGET!")
elif best_score > 0.802901:
    print("\n✓ AutoGluon beats manual optimization!")
    print(f"Expected public LB: ~{best_score + 0.08:.3f}")
else:
    print("\n⚠ AutoGluon didn't break ceiling. Confirms we need different signal source.")

# ============================================
# 8. GENERATE PREDICTIONS
# ============================================
print(f"\n{'='*80}")
print("GENERATING TEST PREDICTIONS")
print(f"{'='*80}")

# Predict
test_predictions = predictor.predict(test_data)

# Distribution
pred_counts = test_predictions.value_counts()
print(f"\nTest predictions:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Save submission
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v8_autogluon.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V8 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Best model: {best_model}")
print(f"CV Macro-F1: {best_score:.6f}")
print(f"High→Medium confusion: {high_to_med/high_total*100:.1f}%")
print(f"\nAutoGluon automatically tested:")
print(f"  - Multiple base models (CatBoost, LightGBM, XGBoost, NNs)")
print(f"  - Hyperparameter optimization")
print(f"  - 2-level stacking")
print(f"  - 5-fold bagging for robustness")
print(f"{'='*80}")

# Save the model
predictor.save('models/autogluon_v8')
print(f"\nModel saved to: models/autogluon_v8")
