import pandas as pd
import numpy as np
from io import StringIO
from .debug_decorators import log_call, log_exceptions, timeit

class DataHandler:
    """Handle all data operations including loading, filtering, and exporting"""
    
    def __init__(self):
        self.data = None
    
    def create_sample_data(self):
        """Create sample dataset for demonstration"""
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
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def load_from_csv(self, uploaded_file):
        """Load data from uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            self.data = df
            return df, None
        except Exception as e:
            return None, str(e)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_data_info(self, df):
        """Get basic information about the dataset"""
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'memory': df.memory_usage(deep=True).sum() / 1024**2,  # MB
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(include=['object']).columns),
            'missing_values': df.isnull().sum().sum()
        }
        return info
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def filter_data(self, df, column, operator, value):
        """Filter dataframe based on conditions"""
        if operator == "equals":
            return df[df[column] == value]
        elif operator == "greater than":
            return df[df[column] > value]
        elif operator == "less than":
            return df[df[column] < value]
        elif operator == "contains":
            return df[df[column].astype(str).str.contains(str(value), case=False, na=False)]
        return df
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def sort_data(self, df, column, ascending=True):
        """Sort dataframe by column"""
        return df.sort_values(by=column, ascending=ascending)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def export_to_csv(self, df):
        """Export dataframe to CSV string"""
        return df.to_csv(index=False)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_column_types(self, df):
        """Get column types categorized"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }