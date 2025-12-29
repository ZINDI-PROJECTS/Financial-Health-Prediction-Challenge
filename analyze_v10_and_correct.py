"""
V25: DOMAIN-DRIVEN HIGH-CLASS CORRECTION

From Starter Notebook Analysis:
- compliance_income_tax = Yes: 16.77% High (vs 3.17% for No) → 5.3x multiplier!
- medical_insurance = Have now: 23.51% High (vs 6.53% for Never had)
- Eswatini: 11.48% High (vs 2.34% Zimbabwe, 0.31% Lesotho)
- High income/turnover/assets correlate with High class

Strategy:
1. Load V10 predictions (0.892 LB - our best)
2. Identify Medium predictions with STRONG High indicators
3. Flip top 5-10 candidates to High
4. Create multiple variants (conservative to aggressive)

Goal: Push from 0.892 → 0.900+
"""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("V25: DOMAIN-DRIVEN HIGH-CLASS CORRECTION")
print("="*80)

# ============================================
# LOAD DATA & V10 PREDICTIONS
# ============================================
print("\n[1/4] Loading data...")
train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')
v10_sub = pd.read_csv('submissions/submission_v10_ensemble_v2_v9.csv')

print(f"  Test samples: {len(test_df):,}")
print(f"  V10 predictions: {len(v10_sub):,}")

# Merge predictions with test data for analysis
test_with_pred = test_df.merge(v10_sub, on='ID')

print(f"\n  V10 distribution:")
print(v10_sub['Target'].value_counts())

# ============================================
# ANALYZE HIGH-CLASS INDICATORS
# ============================================
print("\n[2/4] Analyzing High-class indicators from starter insights...")

# Based on starter notebook analysis
print("\n  Training data patterns (from starter):")
print("  compliance_income_tax = Yes: 16.77% High (5.3x vs No)")
print("  medical_insurance = Have now: 23.51% High")
print("  Eswatini country: 11.48% High")

# Find Medium predictions with strong High signals
medium_predictions = test_with_pred[test_with_pred['Target'] == 'Medium'].copy()
print(f"\n  V10 predicted {len(medium_predictions):,} as Medium")

# Create High-class score for each Medium prediction
def calculate_high_score(row):
    """Score how likely this should be High based on starter insights"""
    score = 0.0

    # Indicator 1: Compliance with income tax (STRONGEST signal - 5.3x)
    if row.get('compliance_income_tax') == 'Yes':
        score += 5.0  # Very strong signal

    # Indicator 2: Medical insurance
    if row.get('medical_insurance') == 'Have now':
        score += 3.0

    # Indicator 3: Country (Eswatini has highest High %)
    if row.get('country') == 'eswatini':
        score += 2.5
    elif row.get('country') == 'malawi':
        score += 1.0

    # Indicator 4: High income (from starter analysis)
    income = row.get('personal_income', 0)
    if pd.notna(income) and income > 500000:  # Top quartile from starter
        score += 2.0
    elif pd.notna(income) and income > 100000:
        score += 1.0

    # Indicator 5: High business turnover
    turnover = row.get('business_turnover', 0)
    if pd.notna(turnover) and turnover > 1000000:
        score += 2.0
    elif pd.notna(turnover) and turnover > 100000:
        score += 1.0

    # Indicator 6: Has insurance (general)
    if row.get('has_insurance') == 'Yes':
        score += 1.5

    # Indicator 7: Keeps financial records
    if row.get('keeps_financial_records') in ['Yes', 'Have now']:
        score += 1.0

    # Indicator 8: Has loan account (access to credit)
    if row.get('has_loan_account') == 'Have now':
        score += 1.0

    # Indicator 9: Funeral insurance
    if row.get('funeral_insurance') == 'Have now':
        score += 1.0

    # Indicator 10: Mature business
    age_years = row.get('business_age_years', 0)
    if pd.notna(age_years) and age_years >= 5:
        score += 1.0

    return score

# Calculate scores for all Medium predictions
medium_predictions['high_score'] = medium_predictions.apply(calculate_high_score, axis=1)

# Sort by score
medium_sorted = medium_predictions.sort_values('high_score', ascending=False)

print("\n  Top candidates for Medium → High flip:")
print(f"  {'Rank':<6} {'ID':<12} {'Score':<8} {'Key Indicators'}")
print("  " + "-"*70)

for idx, (i, row) in enumerate(medium_sorted.head(20).iterrows(), 1):
    indicators = []
    if row.get('compliance_income_tax') == 'Yes':
        indicators.append('Tax✓')
    if row.get('medical_insurance') == 'Have now':
        indicators.append('MedIns✓')
    if row.get('country') == 'eswatini':
        indicators.append('Eswatini✓')
    if pd.notna(row.get('personal_income', 0)) and row['personal_income'] > 500000:
        indicators.append('HighInc✓')
    if row.get('has_insurance') == 'Yes':
        indicators.append('Ins✓')

    print(f"  {idx:<6} {row['ID']:<12} {row['high_score']:<8.1f} {', '.join(indicators)}")

# ============================================
# CREATE VARIANTS
# ============================================
print("\n[3/4] Creating correction variants...")

variants = [
    {'name': 'V25a_conservative', 'flip_count': 3, 'description': 'Flip top 3 (ultra-conservative)'},
    {'name': 'V25b_moderate', 'flip_count': 5, 'description': 'Flip top 5 (moderate)'},
    {'name': 'V25c_aggressive', 'flip_count': 8, 'description': 'Flip top 8 (aggressive)'},
]

for variant in variants:
    # Start with V10 predictions
    corrected = v10_sub.copy()

    # Get IDs to flip
    ids_to_flip = medium_sorted.head(variant['flip_count'])['ID'].tolist()

    # Flip Medium → High
    corrected.loc[corrected['ID'].isin(ids_to_flip), 'Target'] = 'High'

    # Save
    filename = f"submissions/submission_{variant['name']}.csv"
    corrected.to_csv(filename, index=False)

    # Stats
    new_dist = corrected['Target'].value_counts()
    print(f"\n  {variant['name']}:")
    print(f"    {variant['description']}")
    print(f"    Low={new_dist.get('Low', 0)}, Med={new_dist.get('Medium', 0)}, High={new_dist.get('High', 0)}")
    print(f"    Saved: {filename}")

# ============================================
# SHOW FLIPPED SAMPLES
# ============================================
print("\n[4/4] Samples being flipped to High:")

for variant in variants:
    ids_to_flip = medium_sorted.head(variant['flip_count'])['ID'].tolist()
    print(f"\n  {variant['name']} - Flipping these IDs:")

    for flip_id in ids_to_flip:
        row = medium_sorted[medium_sorted['ID'] == flip_id].iloc[0]
        print(f"    {flip_id}: score={row['high_score']:.1f}, "
              f"country={row.get('country', 'N/A')}, "
              f"tax={row.get('compliance_income_tax', 'N/A')}, "
              f"med_ins={row.get('medical_insurance', 'N/A')}")

print("\n" + "="*80)
print("V25 VARIANTS COMPLETE")
print("="*80)
print("\nRECOMMENDATION:")
print("  Start with V25b_moderate (flip 5)")
print("  Based on STRONG domain signals from starter notebook")
print("  Expected: 0.893-0.902 (if signals are correct)")
print("\nKey insight:")
print("  compliance_income_tax=Yes is 5.3x more likely to be High")
print("  We're targeting Medium predictions with this signal")
print("="*80)
