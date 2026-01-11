import pandas as pd
import numpy as np


def create_sample_data() -> pd.DataFrame:
    """Create sample dataset for demonstration and debugging"""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')

    df = pd.DataFrame({
        'date': dates,
        'sales': np.random.randint(100, 1000, 100) + np.random.randn(100) * 50,
        'customers': np.random.randint(10, 100, 100),
        'category': np.random.choice(['Electronics', 'Clothing', 'Food', 'Books'], 100),
        'region': np.random.choice(['North', 'South', 'East', 'West'], 100),
        'profit': np.random.randint(50, 500, 100) + np.random.randn(100) * 25
    })

    df['sales'] = df['sales'].clip(lower=0)
    df['profit'] = df['profit'].clip(lower=0)

    return df
