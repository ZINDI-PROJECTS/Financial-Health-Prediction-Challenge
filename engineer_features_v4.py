"""
V4 Feature Engineering - CLEAN SIGNAL ONLY
Based on V3 results: Keep top performers, remove noise

KEEP:
- credit_access_score (ranked #3, clean)
- profit_margin (ranked #6, refine implementation)

ADD:
- business_maturity (simple, clean interaction)

REMOVE: All noisy features (income_dependency, expense_burden, etc.)
"""
import numpy as np
import pandas as pd


def engineer_features_v4(df, caps=None, is_train=True):
    """
    Add only 3 clean, high-signal features

    Returns:
    - df with new features
    - caps dict (for test set consistency)
    """
    df = df.copy()

    # ========================================
    # FEATURE 1: CREDIT ACCESS SCORE (PROVEN - Rank #3)
    # ========================================
    # Clean ordinal feature: 0 (no access) to 3 (full access)
    credit_features = ['has_credit_card', 'has_loan_account', 'has_debit_card']
    df['credit_access_score'] = 0

    for feat in credit_features:
        # Count "Have now" or "Used to have" as access
        df['credit_access_score'] += (df[feat].isin([
            'Have now',
            'Used to have but don\'t have now'
        ])).astype(int)

    # ========================================
    # FEATURE 2: PROFIT MARGIN (REFINED - Rank #6)
    # ========================================
    # Issue in V3: Extreme outliers from div-by-zero handling
    # Fix: Better handling + log transformation for stability

    # Safe division: only calculate if expenses > 0
    safe_expenses = df['business_expenses'].replace(0, np.nan)
    profit_margin_raw = df['business_turnover'] / safe_expenses

    # Handle infinities and outliers
    if is_train:
        # Use 95th percentile (not 99th) for tighter control
        p95 = profit_margin_raw.quantile(0.95)
        profit_margin_capped = profit_margin_raw.clip(upper=p95)

        # Log transform for stability (handles skewness)
        df['profit_margin'] = np.log1p(profit_margin_capped)

        # Fill NaN with median (businesses with 0 expenses)
        median_profit = df['profit_margin'].median()
        df['profit_margin'] = df['profit_margin'].fillna(median_profit)

        caps_dict = {'profit_margin_p95': p95, 'profit_margin_median': median_profit}

    else:
        # Apply training caps to test
        profit_margin_capped = profit_margin_raw.clip(upper=caps['profit_margin_p95'])
        df['profit_margin'] = np.log1p(profit_margin_capped)
        df['profit_margin'] = df['profit_margin'].fillna(caps['profit_margin_median'])
        caps_dict = caps

    # ========================================
    # FEATURE 3: BUSINESS MATURITY (NEW - Simple & Clean)
    # ========================================
    # Mature businesses with formal record-keeping = High class signal
    # Simple binary multiplication (less noisy than multi-feature sum)

    has_records = df['keeps_financial_records'].isin([
        'Yes, always',
        'Yes, sometimes'
    ]).astype(int)

    # Interaction: age × formalization (0 if informal, age if formal)
    df['business_maturity'] = df['business_age_years'] * has_records

    if is_train:
        return df, caps_dict
    else:
        return df


def prepare_data_v4(train_df, test_df):
    """
    Full pipeline with V4 features
    """
    print("Engineering V4 features (clean signal only)...")

    # Engineer
    train_eng, caps = engineer_features_v4(train_df, is_train=True)
    test_eng = engineer_features_v4(test_df, caps=caps, is_train=False)

    # Separate target
    y = train_eng['Target']
    X = train_eng.drop(['Target', 'ID'], axis=1)
    X_test = test_eng.drop(['ID'], axis=1)
    test_ids = test_eng['ID'].values

    # Feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"  Original features: 37")
    print(f"  Engineered features: 3")
    print(f"  Total features: {X.shape[1]}")
    print(f"  New features: credit_access_score, profit_margin (refined), business_maturity")

    # Handle missing values
    for col in categorical_cols:
        X[col] = X[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)

    for col in numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X, y, X_test, test_ids, categorical_cols


if __name__ == '__main__':
    # Test
    train_df = pd.read_csv('data/raw/Train.csv')
    test_df = pd.read_csv('data/raw/Test.csv')

    X, y, X_test, test_ids, cat_cols = prepare_data_v4(train_df, test_df)

    print("\n" + "="*60)
    print("V4 FEATURE VALIDATION")
    print("="*60)
    print(f"\nX shape: {X.shape}")
    print(f"X_test shape: {X_test.shape}")

    new_features = ['credit_access_score', 'profit_margin', 'business_maturity']
    print(f"\nNew features statistics:")
    for feat in new_features:
        print(f"  {feat}:")
        print(f"    mean={X[feat].mean():.4f}, std={X[feat].std():.4f}")
        print(f"    min={X[feat].min():.4f}, max={X[feat].max():.4f}")
        print(f"    nulls={X[feat].isnull().sum()}")

    print(f"\nSample values (first 5 rows):")
    print(X[new_features].head())
