"""
Ensemble V2 + V9 Predictions

Strategy:
- V2 (0.883 LB): Conservative, excellent generalization
- V9 (0.888 LB): Better High class detection
- Blend: Average probabilities, then predict

Expected: 0.891-0.895 LB
"""
import pandas as pd
import numpy as np

# We don't have the original probability files, so we'll need to
# reload models and regenerate probabilities from V2 and V9 training scripts

# For now, let's create a soft voting ensemble by rerunning both models
# and averaging their probabilities

print("="*80)
print("V10: ENSEMBLE V2 + V9 PREDICTIONS")
print("="*80)

# Load V2 and V9 submissions to analyze
v2_sub = pd.read_csv('submissions/submission_v2_catboost_lgb_blend.csv')
v9_sub = pd.read_csv('submissions/submission_v9_smote_high_class.csv')

print(f"\nV2 predictions:")
print(v2_sub['Target'].value_counts())
print(f"\nV9 predictions:")
print(v9_sub['Target'].value_counts())

# Check if IDs match
assert (v2_sub['ID'] == v9_sub['ID']).all(), "IDs don't match!"

# Since we only have hard predictions (not probabilities), we need to
# regenerate probabilities from both models
# This requires retraining - let me create a proper ensemble script

print("\n" + "="*80)
print("NOTE: To create a true probability ensemble, we need to:")
print("1. Retrain V2 models and save test probabilities")
print("2. Retrain V9 models and save test probabilities")
print("3. Average probabilities and predict")
print("="*80)
print("\nCreating comprehensive ensemble script...")
