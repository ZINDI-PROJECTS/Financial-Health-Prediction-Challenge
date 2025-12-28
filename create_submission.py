"""
Zindi Submission Generator - Financial Health Prediction Challenge
Trains final CatBoost model on 100% of training data and creates submission file
"""

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

# Fix random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("CREATING ZINDI SUBMISSION - FINANCIAL HEALTH PREDICTION")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1/5] Loading Data...")

train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')
sample_submission = pd.read_csv('data/raw/SampleSubmission.csv')

print(f"  Train shape: {train_df.shape}")
print(f"  Test shape: {test_df.shape}")
print(f"  Sample submission shape: {sample_submission.shape}")

# ============================================
# 2. PREPARE FEATURES
# ============================================
print("\n[2/5] Preparing Features...")

# Separate features and target
y = train_df['Target']
X = train_df.drop(['Target', 'ID'], axis=1)
X_test = test_df.drop(['ID'], axis=1)
test_ids = test_df['ID'].values

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"  Categorical features: {len(categorical_cols)}")
print(f"  Numerical features: {len(numerical_cols)}")

# Handle missing values
# Categorical: fill with 'Missing'
for col in categorical_cols:
    X[col] = X[col].fillna('Missing')
    X_test[col] = X_test[col].fillna('Missing')

# Numerical: fill with median
for col in numerical_cols:
    median_val = X[col].median()
    X[col] = X[col].fillna(median_val)
    X_test[col] = X_test[col].fillna(median_val)

# Convert categorical to string for CatBoost
for col in categorical_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

print(f"  Missing values handled")
print(f"  X shape: {X.shape}")
print(f"  X_test shape: {X_test.shape}")

# Encode target
label_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
reverse_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
y_encoded = y.map(label_mapping)

# ============================================
# 3. TRAIN FINAL MODEL
# ============================================
print("\n[3/5] Training Final Model on 100% of Training Data...")

# Initialize CatBoost with same parameters as baseline
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

# Train on full training data
model.fit(
    X,
    y_encoded,
    cat_features=categorical_cols,
    verbose=100
)

print(f"  Model trained successfully")
print(f"  Best iteration: {model.best_iteration_}")
print(f"  Tree count: {model.tree_count_}")

# Save model
model_path = 'models/catboost_final.cbm'
model.save_model(model_path)
print(f"  Model saved to: {model_path}")

# ============================================
# 4. GENERATE PREDICTIONS
# ============================================
print("\n[4/5] Generating Predictions on Test Set...")

# Predict class labels
predictions_encoded = model.predict(X_test).flatten()
predictions = [reverse_mapping[int(pred)] for pred in predictions_encoded]

# Validate predictions
print(f"  Number of predictions: {len(predictions)}")
print(f"  Test set size: {len(test_df)}")
assert len(predictions) == len(test_df), "Prediction count mismatch!"

# Print class distribution
pred_counts = pd.Series(predictions).value_counts().sort_index()
print(f"\n  Predicted class distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = pred_counts.get(cls, 0)
    pct = (count / len(predictions)) * 100
    print(f"    {cls:8s}: {count:5d} ({pct:5.2f}%)")

# ============================================
# 5. CREATE SUBMISSION FILE
# ============================================
print("\n[5/5] Creating Submission File...")

# Create submission dataframe matching sample format
submission = pd.DataFrame({
    'ID': test_ids,
    'Target': predictions
})

# Validate submission format
print(f"\n  Validation checks:")
print(f"    ✓ Columns match: {list(submission.columns) == list(sample_submission.columns)}")
print(f"    ✓ Row count matches: {len(submission) == len(sample_submission)}")
print(f"    ✓ ID order preserved: {all(submission['ID'] == test_ids)}")
print(f"    ✓ Valid labels only: {set(predictions).issubset({'Low', 'Medium', 'High'})}")
print(f"    ✓ No missing values: {submission.isnull().sum().sum() == 0}")

# Save submission file
submission_path = 'submissions/baseline_catboost_v1.csv'
submission.to_csv(submission_path, index=False)

print(f"\n  Submission saved to: {submission_path}")
print(f"  Submission shape: {submission.shape}")

# Display first few rows
print(f"\n  First 10 rows of submission:")
print(submission.head(10))

print("\n" + "="*80)
print("SUBMISSION CREATED SUCCESSFULLY!")
print("="*80)
print(f"\nFinal Summary:")
print(f"  - Model: CatBoostClassifier")
print(f"  - Trained on: {len(train_df):,} samples")
print(f"  - Predictions: {len(submission):,} samples")
print(f"  - File: {submission_path}")
print(f"  - Shape: {submission.shape}")
print("="*80)
