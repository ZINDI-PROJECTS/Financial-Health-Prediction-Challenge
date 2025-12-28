"""
Feature Engineering for High Class Discrimination
Based on EDA: profit_margin is THE critical feature
"""
import numpy as np
import pandas as pd


def engineer_features(df, is_train=True):
    """
    Add 7 engineered features targeting High vs Medium separation

    Key insight from EDA:
    - Raw turnover/expenses overlap heavily between High and Medium
    - Need RATIOS that measure efficiency, sustainability, resilience
    """
    df = df.copy()

    # ========================================
    # TIER 1: EFFICIENCY RATIOS (EDA-VALIDATED)
    # ========================================

    # 1. PROFIT MARGIN (turnover / expenses)
    # EDA showed: High class has better ratio, not just higher absolutes
    df['profit_margin'] = df['business_turnover'] / (df['business_expenses'] + 1)  # +1 to avoid div/0

    # 2. EXPENSE BURDEN (expenses / total income)
    # Measures sustainability: low = sustainable, high = burning cash
    total_income = df['personal_income'] + df['business_turnover'] + 1
    df['expense_burden'] = df['business_expenses'] / total_income

    # 3. REVENUE PER AGE (turnover / business age)
    # Maturity efficiency: established businesses should have stable revenue
    df['revenue_per_age'] = df['business_turnover'] / (df['business_age_years'] + 1)

    # ========================================
    # TIER 2: FINANCIAL ACCESS SCORES
    # ========================================

    # 4. CREDIT ACCESS SCORE (0-3)
    # Financially healthy businesses have access to formal credit
    credit_features = ['has_credit_card', 'has_loan_account', 'has_debit_card']
    df['credit_access_score'] = 0
    for feat in credit_features:
        df['credit_access_score'] += (df[feat].isin(['Have now', 'Used to have but don\'t have now'])).astype(int)

    # 5. FORMALIZATION SCORE (0-3)
    # High class = formal, compliant, insured
    formalization_map = {
        'keeps_financial_records': ['Yes, always', 'Yes, sometimes'],
        'compliance_income_tax': ['Yes'],
        'has_insurance': ['Yes']
    }
    df['formalization_score'] = 0
    for feat, values in formalization_map.items():
        df['formalization_score'] += df[feat].isin(values).astype(int)

    # ========================================
    # TIER 3: STABILITY INTERACTIONS
    # ========================================

    # 6. STABILITY INDEX (age × formalization)
    # Combines longevity with formal practices
    df['stability_index'] = df['business_age_years'] * df['formalization_score']

    # 7. INCOME DEPENDENCY (personal_income / turnover)
    # High = owner heavily dependent on business (risky)
    # Low = business is self-sustaining
    df['income_dependency'] = df['personal_income'] / (df['business_turnover'] + 1)

    # ========================================
    # HANDLE INFINITIES AND OUTLIERS
    # ========================================

    # Cap extreme ratios (99th percentile)
    ratio_features = ['profit_margin', 'expense_burden', 'revenue_per_age', 'income_dependency']

    if is_train:
        caps = {}
        for feat in ratio_features:
            p99 = df[feat].quantile(0.99)
            caps[feat] = p99
            df[feat] = df[feat].clip(upper=p99)
        return df, caps
    else:
        return df

    return df


def prepare_engineered_data(train_df, test_df):
    """
    Full pipeline: engineer features + handle missing values
    """
    print("Engineering features...")

    # Engineer features
    train_eng, caps = engineer_features(train_df, is_train=True)
    test_eng = engineer_features(test_df, is_train=False)

    # Apply caps to test
    ratio_features = ['profit_margin', 'expense_burden', 'revenue_per_age', 'income_dependency']
    for feat in ratio_features:
        if feat in caps:
            test_eng[feat] = test_eng[feat].clip(upper=caps[feat])

    # Separate target
    y = train_eng['Target']
    X = train_eng.drop(['Target', 'ID'], axis=1)
    X_test = test_eng.drop(['ID'], axis=1)
    test_ids = test_eng['ID'].values

    # Identify feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    print(f"  Original features: 37")
    print(f"  Engineered features: 7")
    print(f"  Total features: {X.shape[1]}")
    print(f"  New ratio features: {[f for f in ratio_features if f in X.columns]}")
    print(f"  New score features: credit_access_score, formalization_score, stability_index")

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
    # Test the pipeline
    train_df = pd.read_csv('data/raw/Train.csv')
    test_df = pd.read_csv('data/raw/Test.csv')

    X, y, X_test, test_ids, cat_cols = prepare_engineered_data(train_df, test_df)

    print("\n" + "="*60)
    print("FEATURE ENGINEERING VALIDATION")
    print("="*60)
    print(f"\nX shape: {X.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"\nNew features created:")
    new_features = ['profit_margin', 'expense_burden', 'revenue_per_age',
                    'credit_access_score', 'formalization_score',
                    'stability_index', 'income_dependency']
    for feat in new_features:
        if feat in X.columns:
            print(f"  ✓ {feat}: mean={X[feat].mean():.4f}, std={X[feat].std():.4f}")

    print("\nSample values (first 5 rows):")
    print(X[new_features].head())
