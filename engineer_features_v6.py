"""
V6 FEATURE ENGINEERING: COUNTRY-SPECIFIC INTERACTIONS

Key insight: Financial health definitions vary DRAMATICALLY by country
- Eswatini: 11.5% High (2.4x average)
- Lesotho: 0.3% High (0.06x average)

Strategy:
1. Keep V4 refined features (credit_access, profit_margin, business_maturity)
2. ADD country interactions (country × profit_margin, country × insurance)
3. Test if country-specific patterns unlock the missing signal
"""
import pandas as pd
import numpy as np


def prepare_data_v6(train_df, test_df):
    """
    Prepare features with country-specific interactions

    Returns:
        X, y, X_test, test_ids, categorical_cols
    """
    print("Engineering V6 features (country interactions)...")

    # Start with copies
    train = train_df.copy()
    test = test_df.copy()

    # ============================================
    # V4 REFINED FEATURES (proven winners)
    # ============================================

    # 1. Credit Access Score (rank #3 in V5)
    # Map "Have now" to 1, everything else to 0
    def map_to_binary(series):
        return (series == 'Have now').astype(int)

    train['credit_access_score'] = (
        map_to_binary(train['has_loan_account']) +
        map_to_binary(train['has_credit_card']) +
        map_to_binary(train['has_debit_card']) +
        map_to_binary(train['has_internet_banking'])
    )
    test['credit_access_score'] = (
        map_to_binary(test['has_loan_account']) +
        map_to_binary(test['has_credit_card']) +
        map_to_binary(test['has_debit_card']) +
        map_to_binary(test['has_internet_banking'])
    )

    # 2. Profit Margin (rank #4 in V5) - refined log-scale version
    for df in [train, test]:
        safe_expenses = df['business_expenses'].replace(0, np.nan)
        profit_margin_raw = df['business_turnover'] / safe_expenses
        # Cap at 95th percentile for stability
        p95 = profit_margin_raw.quantile(0.95)
        profit_margin_capped = profit_margin_raw.clip(upper=p95)
        # Log transform
        df['profit_margin'] = np.log1p(profit_margin_capped.fillna(1.0))

    # 3. Business Maturity (total age)
    train['business_maturity'] = (
        train['business_age_years'] * 12 + train['business_age_months']
    )
    test['business_maturity'] = (
        test['business_age_years'] * 12 + test['business_age_months']
    )

    # ============================================
    # V6 NEW: COUNTRY-SPECIFIC INTERACTIONS
    # ============================================

    # Create country dummies for interaction (CatBoost will handle them efficiently)
    # We use binary indicators because interactions need numeric encoding

    for country in ['eswatini', 'zimbabwe', 'malawi', 'lesotho']:
        # Country × Profit Margin
        # This allows model to learn different profit margin thresholds per country
        train[f'profit_margin_x_{country}'] = (
            train['profit_margin'] * (train['country'] == country).astype(int)
        )
        test[f'profit_margin_x_{country}'] = (
            test['profit_margin'] * (test['country'] == country).astype(int)
        )

        # Country × Insurance
        # Insurance importance varies by country
        train[f'has_insurance_x_{country}'] = (
            (train['has_insurance'] == 'Yes').astype(int) *
            (train['country'] == country).astype(int)
        )
        test[f'has_insurance_x_{country}'] = (
            (test['has_insurance'] == 'Yes').astype(int) *
            (test['country'] == country).astype(int)
        )

        # Country × Credit Access
        # Credit infrastructure varies by country
        train[f'credit_access_x_{country}'] = (
            train['credit_access_score'] * (train['country'] == country).astype(int)
        )
        test[f'credit_access_x_{country}'] = (
            test['credit_access_score'] * (test['country'] == country).astype(int)
        )

    # ============================================
    # FEATURE SUMMARY
    # ============================================
    v4_features = ['credit_access_score', 'profit_margin', 'business_maturity']
    v6_interaction_features = [col for col in train.columns if '_x_' in col]

    print(f"  Original features: 37")
    print(f"  V4 refined features: {len(v4_features)}")
    print(f"  V6 country interactions: {len(v6_interaction_features)}")
    print(f"  Total features: {37 + len(v4_features) + len(v6_interaction_features)}")
    print(f"\n  New country interactions:")
    for feat in sorted(v6_interaction_features):
        print(f"    - {feat}")

    # ============================================
    # PREPARE FINAL DATA
    # ============================================

    # Separate target
    y = train['Target']
    test_ids = test['ID']

    # Drop ID and Target
    X = train.drop(['ID', 'Target'], axis=1)
    X_test = test.drop(['ID'], axis=1)

    # Identify categorical columns (all string/object types)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

    # Handle missing values
    # Categorical: fill with 'Missing'
    for col in categorical_cols:
        X[col] = X[col].fillna('Missing').astype(str)
        X_test[col] = X_test[col].fillna('Missing').astype(str)

    # Numerical: fill with median
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X, y, X_test, test_ids, categorical_cols


if __name__ == '__main__':
    # Test the feature engineering
    train_df = pd.read_csv('data/raw/Train.csv')
    test_df = pd.read_csv('data/raw/Test.csv')

    X, y, X_test, test_ids, categorical_cols = prepare_data_v6(train_df, test_df)

    print(f"\n✓ Feature engineering complete!")
    print(f"  X shape: {X.shape}")
    print(f"  X_test shape: {X_test.shape}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Numerical features: {X.shape[1] - len(categorical_cols)}")
