"""
V21: HIGH-CLASS ERROR-TARGETED FEATURES

Strategy:
1. Replicate V10 with CV to identify which High samples are misclassified
2. Analyze patterns in misclassified High samples
3. Engineer TARGETED features specifically for those patterns
4. Retrain V10 approach with new features

Why this might work:
- Surgical approach: fix EXACTLY what V10 gets wrong
- Focus on High class (the hardest, most valuable)
- Data-driven feature engineering

Expected: 0.893-0.900 (targeted improvement)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

sys.path.append('.')
from engineer_features_v6 import prepare_data_v6

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("V21: HIGH-CLASS ERROR-TARGETED FEATURES")
print("="*80)

# ============================================
# LOAD DATA
# ============================================
print("\n[1/6] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping).values

print(f"Features: {X.shape[1]}")
print(f"Samples: {len(X):,}")

# ============================================
# REPLICATE V10 TO FIND ERRORS
# ============================================
print("\n[2/6] Replicating V10 to identify High-class errors...")

N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_predictions = np.zeros(len(X), dtype=int)
class_weights = {0: 1.0, 1: 2.5, 2: 7.0}

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ')

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        class_weights=class_weights,
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )
    cat_model.fit(X_train, y_train, cat_features=categorical_cols, verbose=False)

    # LightGBM
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X[col], X_test[col]])
        le.fit(combined)
        X_train_lgb[col] = le.transform(X_train[col])
        X_val_lgb[col] = le.transform(X_val[col])

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight=class_weights,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train)

    # Blend predictions
    cat_proba = cat_model.predict_proba(X_val)
    lgb_proba = lgb_model.predict_proba(X_val_lgb)
    blend_proba = (cat_proba + lgb_proba) / 2
    oof_predictions[val_idx] = np.argmax(blend_proba, axis=1)

    print("Done")

# ============================================
# ANALYZE HIGH-CLASS ERRORS
# ============================================
print("\n[3/6] Analyzing High-class misclassifications...")

# Find true High samples
high_mask = y_encoded == 2
high_indices = np.where(high_mask)[0]

# Find misclassified High samples
high_errors_mask = high_mask & (oof_predictions != 2)
high_errors_indices = np.where(high_errors_mask)[0]

# Find correctly classified High samples
high_correct_mask = high_mask & (oof_predictions == 2)
high_correct_indices = np.where(high_correct_mask)[0]

print(f"\n  Total High samples: {high_mask.sum()}")
print(f"  Correctly classified: {high_correct_mask.sum()} ({high_correct_mask.sum()/high_mask.sum()*100:.1f}%)")
print(f"  Misclassified: {high_errors_mask.sum()} ({high_errors_mask.sum()/high_mask.sum()*100:.1f}%)")

# Get raw data for analysis
X_errors = train_df.iloc[high_errors_indices]
X_correct = train_df.iloc[high_correct_indices]

print(f"\n  Analyzing patterns in misclassified High samples...")

# Numerical features to analyze
num_features = ['total_income', 'total_turnover', 'total_assets', 'total_expenses',
                'employees_count', 'age_of_business', 'has_loan']

print(f"\n  Feature comparison (Misclassified vs Correct High):")
print(f"  {'Feature':<20} {'Misclassified Mean':<20} {'Correct Mean':<20} {'Difference':<15}")
print(f"  {'-'*75}")

for feat in num_features:
    if feat in X_errors.columns and feat in X_correct.columns:
        error_mean = X_errors[feat].mean()
        correct_mean = X_correct[feat].mean()
        diff = error_mean - correct_mean
        print(f"  {feat:<20} {error_mean:<20.2f} {correct_mean:<20.2f} {diff:<15.2f}")

# ============================================
# ENGINEER TARGETED FEATURES
# ============================================
print("\n[4/6] Engineering targeted features for High-class detection...")

def add_targeted_features(df):
    """Add features that help distinguish misclassified High samples"""
    df_new = df.copy()

    # Feature 1: Business efficiency ratio
    df_new['efficiency_ratio'] = df_new['total_income'] / (df_new['total_expenses'] + 1)

    # Feature 2: Asset turnover
    df_new['asset_turnover'] = df_new['total_turnover'] / (df_new['total_assets'] + 1)

    # Feature 3: Employee productivity
    df_new['productivity_per_employee'] = df_new['total_turnover'] / (df_new['employees_count'] + 1)

    # Feature 4: Income to assets ratio
    df_new['income_asset_ratio'] = df_new['total_income'] / (df_new['total_assets'] + 1)

    # Feature 5: Business maturity score
    df_new['maturity_score'] = df_new['age_of_business'] * df_new['employees_count']

    # Feature 6: Loan utilization
    df_new['loan_burden'] = df_new['has_loan'] * (df_new['total_expenses'] / (df_new['total_income'] + 1))

    # Feature 7: Scale indicator
    df_new['business_scale'] = np.log1p(df_new['total_assets']) * np.log1p(df_new['employees_count'])

    # Feature 8: Financial stability
    df_new['stability_score'] = (df_new['total_income'] - df_new['total_expenses']) / (df_new['total_income'] + 1)

    return df_new

# Apply to train and test
X_enhanced = add_targeted_features(X)
X_test_enhanced = add_targeted_features(X_test)

print(f"  Added 8 targeted features")
print(f"  Total features: {X_enhanced.shape[1]}")

# ============================================
# TRAIN V10 WITH ENHANCED FEATURES
# ============================================
print("\n[5/6] Training V10 approach with enhanced features...")

oof_enhanced = np.zeros(len(X_enhanced), dtype=int)
test_probas_enhanced = np.zeros((len(X_test_enhanced), 3))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_enhanced, y_encoded), 1):
    print(f"  Fold {fold}/{N_FOLDS}...", end=' ')

    X_train, X_val = X_enhanced.iloc[train_idx], X_enhanced.iloc[val_idx]
    y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

    # CatBoost
    cat_model = CatBoostClassifier(
        loss_function='MultiClass',
        class_weights=class_weights,
        iterations=1200,  # Slightly more iterations
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False
    )
    cat_model.fit(X_train, y_train, cat_features=categorical_cols, verbose=False)

    # LightGBM
    X_train_lgb = X_train.copy()
    X_val_lgb = X_val.copy()
    X_test_lgb = X_test_enhanced.copy()

    for col in categorical_cols:
        le = LabelEncoder()
        combined = pd.concat([X_enhanced[col], X_test_enhanced[col]])
        le.fit(combined)
        X_train_lgb[col] = le.transform(X_train[col])
        X_val_lgb[col] = le.transform(X_val[col])
        X_test_lgb[col] = le.transform(X_test_enhanced[col])

    lgb_model = LGBMClassifier(
        objective='multiclass',
        num_class=3,
        class_weight=class_weights,
        n_estimators=1200,
        learning_rate=0.05,
        max_depth=6,
        random_state=RANDOM_STATE,
        verbose=-1
    )
    lgb_model.fit(X_train_lgb, y_train)

    # Blend predictions
    cat_proba = cat_model.predict_proba(X_val)
    lgb_proba = lgb_model.predict_proba(X_val_lgb)
    blend_proba = (cat_proba + lgb_proba) / 2

    oof_enhanced[val_idx] = np.argmax(blend_proba, axis=1)

    # Test predictions
    cat_test_proba = cat_model.predict_proba(X_test_enhanced)
    lgb_test_proba = lgb_model.predict_proba(X_test_lgb)
    test_probas_enhanced += (cat_test_proba + lgb_test_proba) / 2 / N_FOLDS

    print("Done")

# ============================================
# EVALUATE IMPROVEMENT
# ============================================
print("\n[6/6] Evaluating improvement...")

f1_original = f1_score(y_encoded, oof_predictions, average='macro')
f1_enhanced = f1_score(y_encoded, oof_enhanced, average='macro')

print(f"\n  Original V10 OOF F1: {f1_original:.6f}")
print(f"  Enhanced V21 OOF F1: {f1_enhanced:.6f}")
print(f"  Gain: {f1_enhanced - f1_original:+.6f}")

print(f"\n  Per-class F1 comparison:")
f1_orig_per_class = f1_score(y_encoded, oof_predictions, average=None)
f1_enh_per_class = f1_score(y_encoded, oof_enhanced, average=None)

for i, cls in enumerate(['Low', 'Medium', 'High']):
    print(f"    {cls}: {f1_orig_per_class[i]:.4f} → {f1_enh_per_class[i]:.4f} ({f1_enh_per_class[i] - f1_orig_per_class[i]:+.4f})")

# Make test predictions
test_pred_encoded = np.argmax(test_probas_enhanced, axis=1)
test_predictions = [reverse_mapping[p] for p in test_pred_encoded]

# Distribution
pred_counts = pd.Series(test_predictions).value_counts()
print(f"\nTest predictions distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = count / len(test_predictions) * 100
    print(f"  {cls}: {count:,} ({pct:.1f}%)")

# Compare to V10
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')
v10_counts = v10_sub['Target'].value_counts()
print(f"\nComparison to V10:")
print(f"  V10: Low={v10_counts.get('Low', 0)} ({v10_counts.get('Low', 0)/len(v10_sub)*100:.1f}%), "
      f"Med={v10_counts.get('Medium', 0)} ({v10_counts.get('Medium', 0)/len(v10_sub)*100:.1f}%), "
      f"High={v10_counts.get('High', 0)} ({v10_counts.get('High', 0)/len(v10_sub)*100:.1f}%)")
print(f"  V21: Low={pred_counts.get('Low', 0)} ({pred_counts.get('Low', 0)/len(test_predictions)*100:.1f}%), "
      f"Med={pred_counts.get('Medium', 0)} ({pred_counts.get('Medium', 0)/len(test_predictions)*100:.1f}%), "
      f"High={pred_counts.get('High', 0)} ({pred_counts.get('High', 0)/len(test_predictions)*100:.1f}%)")

# Save
submission = pd.DataFrame({'ID': test_ids, 'Target': test_predictions})
filename = 'submissions/submission_v21_error_targeted.csv'
submission.to_csv(filename, index=False)

print(f"\n{'='*80}")
print("V21 COMPLETE")
print(f"{'='*80}")
print(f"File: {filename}")
print(f"Strategy: V10 + Error-Targeted Features for High Class")
print(f"OOF F1 Gain: {f1_enhanced - f1_original:+.6f}")
print(f"Expected LB: 0.893-0.900 (targeted improvement)")
print(f"{'='*80}")
