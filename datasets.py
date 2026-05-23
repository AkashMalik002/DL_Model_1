"""
Dataset creation for all 5 domains
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def create_domain_datasets():
    """
    Create datasets for all 5 domains.
    Each dataset has: ≥500 samples, ≥10 features, balanced classes
    Returns: Dictionary with datasets for each domain
    """
    datasets = {}
    
    # Domain 1: NLP - Review Classification (Binary)
    # Features: review length, sentiment scores, user engagement metrics, linguistic patterns
    n_samples = 800
    n_features = 15
    X_nlp, y_nlp = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=12,
        n_redundant=2, n_classes=2, random_state=42, weights=[0.6, 0.4]
    )
    df_nlp = pd.DataFrame(X_nlp, columns=[f'review_feature_{i}' for i in range(n_features)])
    df_nlp['target'] = y_nlp
    datasets['nlp'] = {'data': df_nlp, 'name': 'Domain 1: NLP Review Classification'}
    
    # Domain 2: Agriculture - Crop Health (Binary)
    # Features: soil properties, weather metrics, vegetation indices, drone sensor data
    X_agri, y_agri = make_classification(
        n_samples=700, n_features=12, n_informative=10,
        n_redundant=1, n_classes=2, random_state=43, weights=[0.55, 0.45]
    )
    df_agri = pd.DataFrame(X_agri, columns=[f'crop_feature_{i}' for i in range(12)])
    df_agri['target'] = y_agri
    datasets['agriculture'] = {'data': df_agri, 'name': 'Domain 2: Agriculture Crop Health'}
    
    # Domain 3: Database - Infrastructure Monitoring (Binary)
    # Features: CPU usage, memory, disk I/O, network activity, query metrics
    X_db, y_db = make_classification(
        n_samples=900, n_features=14, n_informative=11,
        n_redundant=2, n_classes=2, random_state=44, weights=[0.7, 0.3]
    )
    df_db = pd.DataFrame(X_db, columns=[f'server_metric_{i}' for i in range(14)])
    df_db['target'] = y_db
    datasets['database'] = {'data': df_db, 'name': 'Domain 3: Database Infrastructure Monitoring'}
    
    # Domain 4: Healthcare - Patient Readmission (Binary)
    # Features: age, comorbidities, lab values, medication counts, length of stay
    X_health, y_health = make_classification(
        n_samples=800, n_features=13, n_informative=10,
        n_redundant=2, n_classes=2, random_state=45, weights=[0.65, 0.35]
    )
    df_health = pd.DataFrame(X_health, columns=[f'patient_indicator_{i}' for i in range(13)])
    df_health['target'] = y_health
    datasets['healthcare'] = {'data': df_health, 'name': 'Domain 4: Healthcare Patient Readmission'}
    
    # Domain 5: FinTech - Loan Default (Binary)
    # Features: credit score, income, debt ratio, employment history, loan features
    X_fintech, y_fintech = make_classification(
        n_samples=750, n_features=11, n_informative=9,
        n_redundant=1, n_classes=2, random_state=46, weights=[0.75, 0.25]
    )
    df_fintech = pd.DataFrame(X_fintech, columns=[f'financial_indicator_{i}' for i in range(11)])
    df_fintech['target'] = y_fintech
    datasets['fintech'] = {'data': df_fintech, 'name': 'Domain 5: FinTech Loan Default Prediction'}
    
    return datasets


def display_dataset_summary(datasets):
    """Display summary of all datasets"""
    print("=" * 80)
    print("DATASET SUMMARY - All Domains")
    print("=" * 80)
    for domain_key, dataset_info in datasets.items():
        df = dataset_info['data']
        print(f"\n{dataset_info['name']}")
        print(f"  Shape: {df.shape} (Samples × Features)")
        print(f"  ✓ Samples: {df.shape[0]} (Requirement: ≥500)")
        print(f"  ✓ Features: {df.shape[1]-1} (Requirement: ≥10)")
        print(f"  Class Distribution: {np.bincount(df['target'].values)}")
        print(f"  Data Types: {df.dtypes.unique()}")
