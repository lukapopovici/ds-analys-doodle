import pandas as pd
import numpy as np
from scipy import stats

class Analyzer:
    """Perform statistical analysis on data"""
    
    def __init__(self):
        pass
    
    def get_descriptive_stats(self, df, column):
        """Get descriptive statistics for a column"""
        if pd.api.types.is_numeric_dtype(df[column]):
            stats_dict = {
                'Count': df[column].count(),
                'Mean': df[column].mean(),
                'Median': df[column].median(),
                'Mode': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'Std Dev': df[column].std(),
                'Variance': df[column].var(),
                'Min': df[column].min(),
                'Max': df[column].max(),
                'Q1 (25%)': df[column].quantile(0.25),
                'Q3 (75%)': df[column].quantile(0.75),
                'IQR': df[column].quantile(0.75) - df[column].quantile(0.25),
                'Skewness': df[column].skew(),
                'Kurtosis': df[column].kurtosis()
            }
            return pd.DataFrame(stats_dict.items(), columns=['Statistic', 'Value'])
        else:
            # For categorical data
            stats_dict = {
                'Count': df[column].count(),
                'Unique Values': df[column].nunique(),
                'Most Common': df[column].mode().iloc[0] if not df[column].mode().empty else None,
                'Most Common Count': df[column].value_counts().iloc[0] if len(df[column]) > 0 else 0
            }
            return pd.DataFrame(stats_dict.items(), columns=['Statistic', 'Value'])
    
    def get_correlation_matrix(self, df):
        """Get correlation matrix for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def detect_outliers(self, df, column, method='iqr'):
        """Detect outliers in a numeric column"""
        if not pd.api.types.is_numeric_dtype(df[column]):
            return None, "Column must be numeric"
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
            return outliers, f"Found {len(outliers)} outliers using IQR method"
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[column].dropna()))
            outliers = df[z_scores > 3]
            return outliers, f"Found {len(outliers)} outliers using Z-score method"
        
        return None, "Invalid method"
    
    def get_value_counts(self, df, column, top_n=10):
        """Get value counts for a column"""
        counts = df[column].value_counts().head(top_n).reset_index()
        counts.columns = [column, 'Count']
        return counts
    
    def get_missing_values_report(self, df):
        """Get report on missing values"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df)) * 100
        
        report = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Percentage': missing_percent.values
        })
        
        report = report[report['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        return report
    
    def perform_normality_test(self, df, column):
        """Perform Shapiro-Wilk normality test"""
        if not pd.api.types.is_numeric_dtype(df[column]):
            return None, "Column must be numeric"
        
        data = df[column].dropna()
        
        if len(data) < 3:
            return None, "Need at least 3 observations"
        
        if len(data) > 5000:
            # Use a sample for large datasets
            data = data.sample(5000)
        
        statistic, p_value = stats.shapiro(data)
        
        result = {
            'Test': 'Shapiro-Wilk',
            'Statistic': statistic,
            'P-Value': p_value,
            'Normal Distribution': 'Yes' if p_value > 0.05 else 'No',
            'Interpretation': f"The data {'appears to be' if p_value > 0.05 else 'does not appear to be'} normally distributed (α=0.05)"
        }
        
        return pd.DataFrame(result.items(), columns=['Metric', 'Value']), None
    
    def get_group_statistics(self, df, group_col, value_col):
        """Get statistics grouped by a categorical column"""
        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return None, "Value column must be numeric"
        
        grouped = df.groupby(group_col)[value_col].agg([
            ('Count', 'count'),
            ('Mean', 'mean'),
            ('Median', 'median'),
            ('Std Dev', 'std'),
            ('Min', 'min'),
            ('Max', 'max')
        ]).reset_index()
        
        return grouped, None