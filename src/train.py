"""
Training pipeline for Financial Health Prediction
STEP 1: Threshold Tuning Implementation
"""
import sys
import numpy as np
import pandas as pd
from features import prepare_features, get_label_mapping
from model import train_catboost_cv

# Set random seed
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("="*80)
print("STEP 1: THRESHOLD TUNING FOR MACRO-F1 OPTIMIZATION")
print("="*80)

# Load data
print("\nLoading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Prepare features (baseline config for now)
print("\nPreparing features...")
feature_config = {
    'add_missing_indicators': False,
    'log_transform_numerics': False
}

X, y, X_test, test_ids, categorical_cols = prepare_features(
    train_df, test_df, config=feature_config
)

# Encode target
label_mapping = get_label_mapping()
y_encoded = y.map(label_mapping)

print(f"Features shape: {X.shape}")
print(f"Categorical features: {len(categorical_cols)}")

# BASELINE: Auto class weights (what we had before)
print("\n" + "="*80)
print("BASELINE: Auto Class Weights + Threshold Tuning")
print("="*80)

baseline_results = train_catboost_cv(
    X, y_encoded, categorical_cols,
    n_folds=5,
    class_weights=None,  # Auto
    random_state=RANDOM_STATE
)

print("\n" + "="*80)
print("STEP 1 COMPLETE")
print("="*80)
print(f"\nKEY FINDINGS:")
print(f"  Before threshold tuning: {baseline_results['mean_f1_raw']:.6f} ± {baseline_results['std_f1_raw']:.6f}")
print(f"  After threshold tuning:  {baseline_results['mean_f1_tuned']:.6f} ± {baseline_results['std_f1_tuned']:.6f}")
print(f"  Gain from tuning:        {baseline_results['mean_f1_tuned'] - baseline_results['mean_f1_raw']:+.6f}")
print(f"\n  Global optimized thresholds: {baseline_results['global_thresholds']}")
print(f"  Final CV Macro-F1: {baseline_results['f1_global']:.6f}")
