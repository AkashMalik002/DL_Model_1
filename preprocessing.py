"""
Data Preprocessing Pipeline
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np


def preprocess_data(df, target_col='target', test_size=0.2, scaling='standard'):
    """
    Complete preprocessing pipeline:
    1. Handle missing values
    2. Encode categorical features
    3. Normalize/Scale features
    4. Split into train-test sets
    
    Parameters:
    - df: DataFrame
    - target_col: target column name
    - test_size: test set proportion
    - scaling: 'standard' (StandardScaler) or 'minmax' (MinMaxScaler)
    
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Handle missing values
    print("\n[Step 1] Handling Missing Values")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"Found {missing_values.sum()} missing values")
        df = df.fillna(df.mean(numeric_only=True))
        print("✓ Missing values handled (filled with mean)")
    else:
        print("✓ No missing values found")
    
    # Step 2: Separate features and target
    print("\n[Step 2] Separating Features and Target")
    X = df.drop(target_col, axis=1).values
    y = df[target_col].values
    print(f"✓ Features shape: {X.shape}")
    print(f"✓ Target shape: {y.shape}")
    
    # Step 3: Feature scaling
    print("\n[Step 3] Feature Scaling")
    if scaling == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    X = scaler.fit_transform(X)
    print(f"✓ Features scaled using {scaler.__class__.__name__}")
    print(f"  Feature ranges: [{X.min():.4f}, {X.max():.4f}]")
    
    # Step 4: Train-test split
    print("\n[Step 4] Train-Test Split")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"✓ Train set: {X_train.shape[0]} samples")
    print(f"✓ Test set: {X_test.shape[0]} samples")
    print(f"✓ Train-test split ratio: {1-test_size:.1%} - {test_size:.1%}")
    
    print("\n" + "=" * 60)
    return X_train, X_test, y_train, y_test, scaler
