"""
Feature engineering utilities
"""
import numpy as np
import pandas as pd


def prepare_features(train_df, test_df, config=None):
    """
    Prepare features for modeling

    Parameters:
    -----------
    train_df : DataFrame
        Training data
    test_df : DataFrame
        Test data
    config : dict, optional
        Feature engineering configuration
        - add_missing_indicators: bool
        - log_transform_numerics: bool
        - missing_indicator_cols: list

    Returns:
    --------
    X_train, y_train, X_test, test_ids, categorical_cols
    """
    if config is None:
        config = {}

    # Separate target and features
    y = train_df['Target']
    X = train_df.drop(['Target', 'ID'], axis=1)
    X_test = test_df.drop(['ID'], axis=1)
    test_ids = test_df['ID'].values

    # Identify feature types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # STEP 3: Add missing indicators if requested
    if config.get('add_missing_indicators', False):
        missing_cols = config.get('missing_indicator_cols', [])
        for col in missing_cols:
            if col in X.columns:
                X[f'{col}_is_missing'] = X[col].isnull().astype(int)
                X_test[f'{col}_is_missing'] = X_test[col].isnull().astype(int)

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

    # STEP 4: Log transform heavy-tailed numerics if requested
    if config.get('log_transform_numerics', False):
        log_cols = config.get('log_transform_cols', [])
        for col in log_cols:
            if col in numerical_cols:
                X[f'{col}_log1p'] = np.log1p(X[col])
                X_test[f'{col}_log1p'] = np.log1p(X_test[col])

    # Convert categorical to string
    for col in categorical_cols:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    return X, y, X_test, test_ids, categorical_cols


def get_label_mapping():
    """Get label encoding mapping"""
    return {'Low': 0, 'Medium': 1, 'High': 2}


def get_reverse_mapping():
    """Get reverse label mapping"""
    return {0: 'Low', 1: 'Medium', 2: 'High'}
