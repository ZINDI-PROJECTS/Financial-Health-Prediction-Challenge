"""
Financial Health Prediction Challenge - Baseline Model
Target: macro F1 score optimization
Model: CatBoostClassifier with 5-fold Stratified CV
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Fix random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*60)
print("FINANCIAL HEALTH PREDICTION - BASELINE MODEL")
print("="*60)

# ============================================
# TASK 1: DATA LOADING
# ============================================
print("\n[TASK 1] Loading Data...")

# Load datasets
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Separate features and target
y = train_df['Target']
X = train_df.drop(['Target', 'ID'], axis=1)
X_test = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Print missing values summary
print("\n" + "-"*60)
print("MISSING VALUES SUMMARY")
print("-"*60)
missing_train = X.isnull().sum()
missing_pct = (missing_train / len(X)) * 100
missing_df = pd.DataFrame({
    'Column': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Pct': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)

if len(missing_df) > 0:
    print(missing_df.to_string(index=False))
else:
    print("No missing values found!")

# Print class distribution
print("\n" + "-"*60)
print("CLASS DISTRIBUTION")
print("-"*60)
class_dist = y.value_counts().sort_index()
class_pct = (class_dist / len(y)) * 100

for cls in class_dist.index:
    print(f"{cls:8s}: {class_dist[cls]:5d} ({class_pct[cls]:5.2f}%)")

# ============================================
# TASK 2: FEATURE HANDLING
# ============================================
print("\n[TASK 2] Feature Handling...")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical features ({len(numerical_cols)}): {len(numerical_cols)} columns")

# Handle missing values
# For categorical columns: fill with 'Missing'
for col in categorical_cols:
    X[col] = X[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')

# For numerical columns: fill with median
for col in numerical_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

print("\nMissing values handled:")
print(f"  Categorical: filled with 'Missing'")
print(f"  Numerical: filled with median")

# Convert categorical columns to string for CatBoost
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print("\nFeature types prepared for CatBoost (native categorical support)")

# ============================================
# TASK 3: BASELINE MODEL - CatBoost
# ============================================
print("\n[TASK 3] Training Baseline Model...")
print("\nModel: CatBoostClassifier")
print("CV Strategy: 5-fold Stratified CV")
print("Optimization Metric: Macro F1\n")

# Encode target labels to integers for CatBoost
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
y_encoded = y.map(label_mapping)

# Initialize Stratified K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Storage for results
fold_f1_scores = []
all_y_true = []
all_y_pred = []

# Cross-validation loop
for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y_encoded), 1):
    print(f"\n{'='*60}")
    print(f"FOLD {fold_idx}/5")
    print(f"{'='*60}")

    # Split data
    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]

    # Initialize CatBoost model
    model = CatBoostClassifier(
        loss_function='MultiClass',
        eval_metric='TotalF1',
        auto_class_weights='Balanced',
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_state=RANDOM_STATE,
        verbose=False,
        early_stopping_rounds=50
    )

    # Train model
    model.fit(
        X_train_fold,
        y_train_fold,
        cat_features=categorical_cols,
        eval_set=(X_val_fold, y_val_fold),
        verbose=False
    )

    # Predict on validation set
    y_pred_fold = model.predict(X_val_fold).flatten()

    # Calculate macro F1 for this fold
    fold_f1 = f1_score(y_val_fold, y_pred_fold, average='macro')
    fold_f1_scores.append(fold_f1)

    # Store for aggregated confusion matrix
    all_y_true.extend(y_val_fold.tolist())
    all_y_pred.extend(y_pred_fold.tolist())

    print(f"Validation Macro F1: {fold_f1:.6f}")
    print(f"Best Iteration: {model.best_iteration_}")

# ============================================
# FINAL RESULTS
# ============================================
print("\n" + "="*60)
print("BASELINE EVALUATION RESULTS")
print("="*60)

mean_f1 = np.mean(fold_f1_scores)
std_f1 = np.std(fold_f1_scores)

print(f"\nCross-Validation Macro F1 Scores:")
for i, score in enumerate(fold_f1_scores, 1):
    print(f"  Fold {i}: {score:.6f}")

print(f"\n{'='*60}")
print(f"Mean CV Macro F1: {mean_f1:.6f} ± {std_f1:.6f}")
print(f"{'='*60}")

# Aggregated confusion matrix
print("\nAggregated Confusion Matrix:")
cm = confusion_matrix(all_y_true, all_y_pred)
print("\n         Predicted")
print("         Low  Medium  High")
for i, true_label in enumerate(['Low', 'Medium', 'High']):
    print(f"Actual {true_label:6s}  {cm[i][0]:4d}   {cm[i][1]:4d}   {cm[i][2]:4d}")

# Per-class metrics
print("\nPer-Class Metrics:")
print(classification_report(
    all_y_true,
    all_y_pred,
    target_names=['Low', 'Medium', 'High'],
    digits=4
))

print("\n" + "="*60)
print("BASELINE COMPLETE - STOPPING AS PER INSTRUCTIONS")
print("="*60)
