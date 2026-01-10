import pandas as pd
import numpy as np
from io import StringIO
import multiprocessing as mp
from functools import partial
from typing import Optional, Tuple, Dict, List
import dask.dataframe as dd
import tempfile
import os
            
from .debug_decorators import log_call, log_exceptions, timeit

class DataHandler:
    """Handle all data operations including loading, filtering, and exporting with parallel processing support"""
    """The parallel processing and chunking adds unecessary overhead for small files (makeit automatic?)"""
    def __init__(self):
        self.data = None
        self.use_dask = False
        self.chunk_size = 50000  # default chunk size for reading
    
    def create_sample_data(self):
        """Create sample dataset for demonstration and debugging moronic code"""
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
    
    # I can't necesarily get the size without reading the rows
    # so this is an estimation based on file seek-ing
    def _estimate_file_size(self, uploaded_file) -> int:
        """Estimate file size in bytes"""
        uploaded_file.seek(0, 2)  # Seek to end
        size = uploaded_file.tell()
        uploaded_file.seek(0)  # reset to start
        return size
    
    def _process_chunk(self, chunk: pd.DataFrame, dtypes: Optional[Dict] = None) -> pd.DataFrame:
        """Process a single chunk of data"""
        if dtypes:
            for col, dtype in dtypes.items():
                if col in chunk.columns:
                    try:
                        chunk[col] = chunk[col].astype(dtype)
                    except:
                        pass
        return chunk
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def load_from_csv(self, uploaded_file, use_parallel: bool = None, 
                     chunk_size: Optional[int] = None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Load data from uploaded CSV file with automatic parallel processing for large files
        
        Args:
            uploaded_file: Uploaded file object
            use_parallel: Force parallel processing (None = auto-detect based on size)
            chunk_size: Size of chunks for reading (default: 50000 rows)
        
        Returns:
            Tuple of (DataFrame, error_message)
        """
        try:
            # Estimate (with emphasis on estimation) file size
            file_size_mb = self._estimate_file_size(uploaded_file) / (1024 ** 2)
            
            # Auto-detect if we should use parallel processing (files around 100MB)
            if use_parallel is None:
                use_parallel = file_size_mb > 100
            
            chunk_size = chunk_size or self.chunk_size
            
            if use_parallel:
                return self._load_csv_parallel(uploaded_file, chunk_size, file_size_mb)
            else:
                return self._load_csv_standard(uploaded_file)
                
        except Exception as e:
            return None, str(e)
    
    def _load_csv_standard(self, uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Standard CSV loading for smaller files"""
        try:
            df = pd.read_csv(uploaded_file, low_memory=False)
            self.data = df
            self.use_dask = False
            return df, None
        except Exception as e:
            return None, str(e)
    
    def _load_csv_parallel(self, uploaded_file, chunk_size: int, 
                          file_size_mb: float) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Parallel CSV loading for large files using chunked reading"""
        try:
            uploaded_file.seek(0)
            
            # First pass: read a sample to infer dtypes
            sample_df = pd.read_csv(uploaded_file, nrows=1000)
            dtypes = sample_df.dtypes.to_dict()
            
            # Reset file pointer
            uploaded_file.seek(0)
            
            # Read in chunks and process in parallel
            chunks = []
            n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free
            
            # Create process-pool
            with mp.Pool(processes=n_cores) as pool:
                chunk_iterator = pd.read_csv(
                    uploaded_file, 
                    chunksize=chunk_size,
                    low_memory=False
                )
                
                # Process chunks in parallel
                process_func = partial(self._process_chunk, dtypes=dtypes)
                chunks = pool.map(process_func, chunk_iterator)
            
            # Concatenate all chunks
            df = pd.concat(chunks, ignore_index=True)
            
            # Optimize memory usage
            df = self._optimize_dtypes(df)
            
            self.data = df
            self.use_dask = False
            
            return df, None
            
        except Exception as e:
            return None, f"Parallel loading error: {str(e)}"
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def load_from_csv_dask(self, uploaded_file, blocksize: str = "64MB") -> Tuple[Optional[dd.DataFrame], Optional[str]]:
        """
        Load very large CSV files using Dask for out-of-core processing
        
        Args:
            uploaded_file: Uploaded file object
            blocksize: Size of blocks for Dask (e.g., "64MB", "128MB")
        
        Returns:
            Tuple of (Dask DataFrame, error_message)
        """
        try:
            uploaded_file.seek(0)

            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                ddf = dd.read_csv(tmp_path, blocksize=blocksize)
                #convert to pandas for preview (first partition only)
                preview_df = ddf.head(1000)
                
                self.data = preview_df  # store 4 display
                self.use_dask = True
                self._dask_data = ddf  
                
                return preview_df, None
                
            finally:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            return None, f"Dask loading error: {str(e)}"
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
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
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_data_info(self, df: pd.DataFrame) -> Dict:
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
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def filter_data(self, df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
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
    def filter_data_parallel(self, df: pd.DataFrame, filters: List[Tuple[str, str, any]]) -> pd.DataFrame:
        """Apply multiple filters in parallel for large datasets"""
        if len(filters) == 0:
            return df
        
        # For single filter, use standard method
        if len(filters) == 1:
            return self.filter_data(df, *filters[0])
        
        # Split dataframe into chunks
        n_cores = max(1, mp.cpu_count() - 1)
        chunk_size = len(df) // n_cores
        chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        
        # Apply filters to each chunk in parallel
        with mp.Pool(processes=n_cores) as pool:
            filter_func = partial(self._apply_filters_to_chunk, filters=filters)
            filtered_chunks = pool.map(filter_func, chunks)
        
        return pd.concat(filtered_chunks, ignore_index=True)
    
    def _apply_filters_to_chunk(self, chunk: pd.DataFrame, filters: List[Tuple[str, str, any]]) -> pd.DataFrame:
        """Apply multiple filters to a chunk"""
        result = chunk.copy()
        for column, operator, value in filters:
            result = self.filter_data(result, column, operator, value)
        return result
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def sort_data(self, df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        """Sort dataframe by column"""
        return df.sort_values(by=column, ascending=ascending)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def export_to_csv(self, df: pd.DataFrame, chunk_export: bool = False) -> str:
        """
        Export dataframe to CSV string
        
        Args:
            df: DataFrame to export
            chunk_export: Use chunked export for large datasets
        """
        if chunk_export and len(df) > 100000:
            
            return df.to_csv(index=False, chunksize=50000)
        return df.to_csv(index=False)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get column types categorized"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        return {
            'numeric': numeric_cols,
            'categorical': categorical_cols,
            'datetime': datetime_cols
        }
    
    def get_memory_usage_report(self, df: pd.DataFrame) -> Dict:
        """Get detailed memory usage report"""
        memory_usage = df.memory_usage(deep=True)
        
        return {
            'total_mb': memory_usage.sum() / 1024**2,
            'per_column': {
                col: memory_usage[col] / 1024**2 
                for col in df.columns
            },
            'optimized': self._can_optimize_memory(df)
        }
    
    def _can_optimize_memory(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame can benefit from memory optimization"""
        # check for int64/float64 datatypes that could be downcasted
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            return True
        
        # check for object columns with low cardinality
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df[col]) < 0.5:
                return True
        
        return False