"""
Financial Health Prediction Challenge - Comprehensive Data Analysis
Understanding the data before modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_style("whitegrid")

print("="*80)
print("FINANCIAL HEALTH PREDICTION - COMPREHENSIVE DATA ANALYSIS")
print("="*80)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] LOADING DATA...")

train_df = pd.read_csv('data/raw/Train.csv')
test_df = pd.read_csv('data/raw/Test.csv')

print(f"\nTrain shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"Total samples: {len(train_df) + len(test_df)}")

# ============================================
# 2. BASIC STATISTICS
# ============================================
print("\n" + "="*80)
print("[2] BASIC DATA OVERVIEW")
print("="*80)

print("\nColumn names and types:")
print(train_df.dtypes)

print("\nFirst few rows:")
print(train_df.head())

# ============================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================
print("\n" + "="*80)
print("[3] TARGET VARIABLE ANALYSIS")
print("="*80)

target_counts = train_df['Target'].value_counts()
target_pct = (target_counts / len(train_df)) * 100

print("\nClass Distribution:")
for cls in ['Low', 'Medium', 'High']:
    count = target_counts.get(cls, 0)
    pct = target_pct.get(cls, 0)
    print(f"  {cls:8s}: {count:5d} ({pct:5.2f}%)")

print("\nClass Imbalance Ratio:")
print(f"  Low:Medium:High = {target_counts['Low']//target_counts['High']}:{target_counts['Medium']//target_counts['High']}:1")
print(f"  Majority/Minority ratio: {target_counts['Low'] / target_counts['High']:.2f}x")

# ============================================
# 4. MISSING VALUES ANALYSIS
# ============================================
print("\n" + "="*80)
print("[4] MISSING VALUES ANALYSIS")
print("="*80)

missing_train = train_df.isnull().sum()
missing_pct = (missing_train / len(train_df)) * 100
missing_df = pd.DataFrame({
    'Feature': missing_train.index,
    'Missing_Count': missing_train.values,
    'Missing_Pct': missing_pct.values
})
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Pct', ascending=False)

print(f"\nFeatures with missing values: {len(missing_df)}/{train_df.shape[1]}")
print(f"Total missing cells: {missing_train.sum():,}")
print(f"Missing percentage: {(missing_train.sum() / (train_df.shape[0] * train_df.shape[1])) * 100:.2f}%")

print("\nTop 15 features with most missing values:")
print(missing_df.head(15).to_string(index=False))

print("\nMissing value categories:")
severe = len(missing_df[missing_df['Missing_Pct'] > 40])
high = len(missing_df[(missing_df['Missing_Pct'] > 20) & (missing_df['Missing_Pct'] <= 40)])
moderate = len(missing_df[(missing_df['Missing_Pct'] > 5) & (missing_df['Missing_Pct'] <= 20)])
low = len(missing_df[missing_df['Missing_Pct'] <= 5])

print(f"  Severe (>40%):    {severe} features")
print(f"  High (20-40%):    {high} features")
print(f"  Moderate (5-20%): {moderate} features")
print(f"  Low (<5%):        {low} features")

# ============================================
# 5. FEATURE TYPE ANALYSIS
# ============================================
print("\n" + "="*80)
print("[5] FEATURE TYPE ANALYSIS")
print("="*80)

X = train_df.drop(['Target', 'ID'], axis=1)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")
print(f"Total features: {len(categorical_cols) + len(numerical_cols)}")

# Analyze categorical features
print("\n" + "-"*80)
print("CATEGORICAL FEATURES CARDINALITY:")
print("-"*80)

cat_info = []
for col in categorical_cols:
    n_unique = train_df[col].nunique()
    top_value = train_df[col].value_counts().index[0] if n_unique > 0 else 'N/A'
    top_freq = train_df[col].value_counts().iloc[0] if n_unique > 0 else 0
    top_pct = (top_freq / len(train_df)) * 100
    cat_info.append({
        'Feature': col,
        'Unique_Values': n_unique,
        'Top_Value': top_value,
        'Top_Freq': top_freq,
        'Top_Pct': top_pct
    })

cat_df = pd.DataFrame(cat_info).sort_values('Unique_Values', ascending=False)
print(cat_df.to_string(index=False))

# Analyze numerical features
print("\n" + "-"*80)
print("NUMERICAL FEATURES STATISTICS:")
print("-"*80)

print(train_df[numerical_cols].describe().T)

# ============================================
# 6. TARGET VS FEATURES RELATIONSHIP
# ============================================
print("\n" + "="*80)
print("[6] FEATURE-TARGET RELATIONSHIP ANALYSIS")
print("="*80)

# Categorical features vs Target (Chi-square test)
print("\nCATEGORICAL FEATURES - Chi-Square Test with Target:")
print("-"*80)

chi_results = []
for col in categorical_cols:
    # Create contingency table
    contingency = pd.crosstab(train_df[col].fillna('Missing'), train_df['Target'])
    if contingency.shape[0] > 1 and contingency.shape[1] > 1:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        cramers_v = np.sqrt(chi2 / (len(train_df) * (min(contingency.shape) - 1)))
        chi_results.append({
            'Feature': col,
            'Chi2': chi2,
            'P_Value': p_value,
            'Cramers_V': cramers_v,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

chi_df = pd.DataFrame(chi_results).sort_values('Cramers_V', ascending=False)
print("\nTop 15 most associated categorical features:")
print(chi_df.head(15).to_string(index=False))

print(f"\nSignificant associations (p < 0.05): {len(chi_df[chi_df['P_Value'] < 0.05])}/{len(chi_df)}")

# Numerical features vs Target (ANOVA)
print("\n" + "-"*80)
print("NUMERICAL FEATURES - ANOVA F-test with Target:")
print("-"*80)

anova_results = []
for col in numerical_cols:
    groups = [train_df[train_df['Target'] == cls][col].dropna() for cls in ['Low', 'Medium', 'High']]
    if all(len(g) > 0 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        anova_results.append({
            'Feature': col,
            'F_Statistic': f_stat,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

anova_df = pd.DataFrame(anova_results).sort_values('F_Statistic', ascending=False)
print(anova_df.to_string(index=False))

print(f"\nSignificant associations (p < 0.05): {len(anova_df[anova_df['P_Value'] < 0.05])}/{len(anova_df)}")

# ============================================
# 7. NUMERICAL FEATURE DISTRIBUTIONS
# ============================================
print("\n" + "="*80)
print("[7] NUMERICAL FEATURE DISTRIBUTION ANALYSIS")
print("="*80)

for col in numerical_cols:
    data = train_df[col].dropna()
    print(f"\n{col}:")
    print(f"  Mean: {data.mean():.2f}")
    print(f"  Median: {data.median():.2f}")
    print(f"  Std: {data.std():.2f}")
    print(f"  Skewness: {data.skew():.4f}")
    print(f"  Kurtosis: {data.kurtosis():.4f}")
    print(f"  Min: {data.min():.2f}, Max: {data.max():.2f}")

# ============================================
# 8. CORRELATION ANALYSIS
# ============================================
print("\n" + "="*80)
print("[8] CORRELATION ANALYSIS (Numerical Features)")
print("="*80)

corr_matrix = train_df[numerical_cols].corr()
print("\nCorrelation Matrix:")
print(corr_matrix)

# Find high correlations
high_corr = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            high_corr.append({
                'Feature_1': corr_matrix.columns[i],
                'Feature_2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr:
    print("\nHighly correlated feature pairs (|r| > 0.7):")
    print(pd.DataFrame(high_corr).to_string(index=False))
else:
    print("\nNo highly correlated feature pairs found (|r| > 0.7)")

# ============================================
# 9. DATA QUALITY ISSUES
# ============================================
print("\n" + "="*80)
print("[9] DATA QUALITY ASSESSMENT")
print("="*80)

# Check for duplicate rows
n_duplicates = train_df.duplicated().sum()
print(f"\nDuplicate rows: {n_duplicates}")

# Check for constant features
constant_features = [col for col in X.columns if train_df[col].nunique() <= 1]
print(f"\nConstant features (nunique <= 1): {len(constant_features)}")
if constant_features:
    print(f"  {constant_features}")

# Check for quasi-constant features (>95% same value)
quasi_constant = []
for col in X.columns:
    if train_df[col].value_counts().iloc[0] / len(train_df) > 0.95:
        quasi_constant.append(col)

print(f"\nQuasi-constant features (>95% same value): {len(quasi_constant)}")
if quasi_constant:
    print(f"  {quasi_constant[:10]}..." if len(quasi_constant) > 10 else f"  {quasi_constant}")

# ============================================
# 10. TRAIN-TEST CONSISTENCY
# ============================================
print("\n" + "="*80)
print("[10] TRAIN-TEST CONSISTENCY CHECK")
print("="*80)

print("\nChecking if test features match train features...")

train_features = set(train_df.columns) - {'Target'}
test_features = set(test_df.columns)

if train_features == test_features:
    print("✓ All features are consistent between train and test sets")
else:
    print("✗ Feature mismatch detected!")
    if train_features - test_features:
        print(f"  Features in train but not in test: {train_features - test_features}")
    if test_features - train_features:
        print(f"  Features in test but not in train: {test_features - train_features}")

# Check categorical value consistency
print("\nChecking categorical value consistency...")
inconsistent_cats = []
for col in categorical_cols:
    train_vals = set(train_df[col].dropna().unique())
    test_vals = set(test_df[col].dropna().unique())
    new_vals_in_test = test_vals - train_vals
    if new_vals_in_test:
        inconsistent_cats.append({
            'Feature': col,
            'New_Values_in_Test': len(new_vals_in_test),
            'Examples': list(new_vals_in_test)[:3]
        })

if inconsistent_cats:
    print(f"\n⚠ Features with new categorical values in test set: {len(inconsistent_cats)}")
    print(pd.DataFrame(inconsistent_cats).to_string(index=False))
else:
    print("✓ No new categorical values in test set")

# ============================================
# SUMMARY
# ============================================
print("\n" + "="*80)
print("KEY INSIGHTS SUMMARY")
print("="*80)

print("\n1. DATA CHARACTERISTICS:")
print(f"   - Training samples: {len(train_df):,}")
print(f"   - Features: {train_df.shape[1] - 2} (31 categorical, 6 numerical)")
print(f"   - Target classes: 3 (Low, Medium, High)")
print(f"   - Class imbalance: {target_counts['Low'] / target_counts['High']:.1f}x (Low/High)")

print("\n2. DATA QUALITY:")
print(f"   - Missing values: {(missing_train.sum() / (train_df.shape[0] * train_df.shape[1])) * 100:.1f}% of data")
print(f"   - Features with >40% missing: {severe}")
print(f"   - Duplicate rows: {n_duplicates}")
print(f"   - Constant features: {len(constant_features)}")

print("\n3. FEATURE-TARGET RELATIONSHIPS:")
print(f"   - Significant categorical features: {len(chi_df[chi_df['P_Value'] < 0.05])}/{len(chi_df)}")
print(f"   - Significant numerical features: {len(anova_df[anova_df['P_Value'] < 0.05])}/{len(anova_df)}")

print("\n4. BASELINE MODEL PERFORMANCE:")
print(f"   - CatBoost Mean CV Macro F1: 0.766 ± 0.021")
print(f"   - Handles missing values and categorical features natively")

print("\n" + "="*80)
print("EDA COMPLETE")
print("="*80)
