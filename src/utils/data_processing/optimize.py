import pandas as pd
import numpy as np
from typing import Dict


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by downcasting numeric types"""
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')

    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')

    # check uniqueness
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df[col])
        if num_unique / num_total < 0.5:  # less than 50% unique values
            df[col] = df[col].astype('category')

    return df


def can_optimize_memory(df: pd.DataFrame) -> bool:
    """Check if DataFrame can benefit from memory optimization"""
    # check for int64/float64 datatypes that could be downcasted
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        return True

    # check for object columns with low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df[col]) < 0.5:
            return True

    return False


def get_memory_usage_report(df: pd.DataFrame) -> Dict:
    """Get detailed memory usage report"""
    memory_usage = df.memory_usage(deep=True)

    return {
        'total_mb': memory_usage.sum() / 1024**2,
        'per_column': {
            col: memory_usage[col] / 1024**2
            for col in df.columns
        },
        'optimized': can_optimize_memory(df)
    }


def get_column_types(df: pd.DataFrame) -> Dict[str, list]:
    """Get column types categorized"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()

    return {
        'numeric': numeric_cols,
        'categorical': categorical_cols,
        'datetime': datetime_cols
    }


def get_data_info(df: pd.DataFrame) -> Dict:
    """Get basic information about the dataset"""
    info = {
        'rows': len(df),
        'columns': len(df.columns),
        'memory': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': len(df.select_dtypes(include=['object', 'category']).columns),
        'missing_values': df.isnull().sum().sum()
    }
    return info
