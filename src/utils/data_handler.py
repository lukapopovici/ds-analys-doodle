import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List
import dask.dataframe as dd
from .debug_decorators import log_call, log_exceptions, timeit
from . import data_processing as dproc

class DataHandler:
    """Handle all data operations including loading, filtering, and exporting with parallel processing support"""
    """The parallel processing and chunking adds unecessary overhead for small files (makeit automatic?)"""
    def __init__(self):
        self.data = None
        self.use_dask = False
        self.chunk_size = 50000  # default chunk size for reading
    
    def create_sample_data(self):
        """Create sample dataset for demonstration and debugging"""
        return dproc.create_sample_data()
    
    # I can't necesarily get the size without reading the rows
    # so this is an estimation based on file seek-ing
    def _estimate_file_size(self, uploaded_file) -> int:
        """Estimate file size in bytes"""
        return dproc.estimate_file_size(uploaded_file)
    
    def _process_chunk(self, chunk: pd.DataFrame, dtypes: Optional[Dict] = None) -> pd.DataFrame:
        """Process a single chunk of data"""
        return dproc.process_chunk(chunk, dtypes=dtypes)
    
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
                df, err = dproc.load_csv_parallel(uploaded_file, chunk_size, file_size_mb, optimize_func=self._optimize_dtypes)
                if df is not None:
                    self.data = df
                    self.use_dask = False
                return df, err
            else:
                df, err = dproc.load_csv_standard(uploaded_file)
                if df is not None:
                    self.data = df
                    self.use_dask = False
                return df, err
                
        except Exception as e:
            return None, str(e)
    
    # CSV loading implementations moved to `utils.data_processing.io`.
    # The `load_from_csv` method delegates to those helpers and updates state (self.data/use_dask).
    
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
            Tuple of (preview pandas DataFrame, error_message)
        """
        try:
            uploaded_file.seek(0)

            preview_df, ddf, err = dproc.load_csv_dask(uploaded_file, blocksize=blocksize)
            if err:
                return None, err

            self.data = preview_df  # store for display
            self.use_dask = True
            self._dask_data = ddf

            return preview_df, None

        except Exception as e:
            return None, f"Dask loading error: {str(e)}"
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        return dproc.optimize_dtypes(df)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_data_info(self, df: pd.DataFrame) -> Dict:
        """Get basic information about the dataset"""
        return dproc.get_data_info(df)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def filter_data(self, df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
        """Filter dataframe based on conditions"""
        return dproc.filter_data(df, column, operator, value)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def filter_data_parallel(self, df: pd.DataFrame, filters: List[Tuple[str, str, any]]) -> pd.DataFrame:
        """Apply multiple filters in parallel for large datasets"""
        return dproc.filter_data_parallel(df, filters)
    
    # Chunk-level filter implementation moved to `utils.data_processing.filtering` module.
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def sort_data(self, df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
        """Sort dataframe by column"""
        return dproc.sort_data(df, column, ascending=ascending)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def export_to_csv(self, df: pd.DataFrame, chunk_export: bool = False) -> str:
        """
        Export dataframe to CSV string
        
        Args:
            df: DataFrame to export
            chunk_export: Use chunked export for large datasets
        """
        return dproc.export_to_csv(df, chunk_export=chunk_export)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def get_column_types(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Get column types categorized"""
        return dproc.get_column_types(df)
    
    def get_memory_usage_report(self, df: pd.DataFrame) -> Dict:
        """Get detailed memory usage report"""
        return dproc.get_memory_usage_report(df)
    
    def _can_optimize_memory(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame can benefit from memory optimization"""
        return dproc.can_optimize_memory(df)