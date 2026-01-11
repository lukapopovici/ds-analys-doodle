import pandas as pd
import multiprocessing as mp
from functools import partial
from typing import Optional, Dict, Tuple
import tempfile
import os
import dask.dataframe as dd


def estimate_file_size(uploaded_file) -> int:
    """Estimate file size in bytes"""
    uploaded_file.seek(0, 2)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    return size


def process_chunk(chunk: pd.DataFrame, dtypes: Optional[Dict] = None) -> pd.DataFrame:
    """Process a single chunk of data"""
    if dtypes:
        for col, dtype in dtypes.items():
            if col in chunk.columns:
                try:
                    chunk[col] = chunk[col].astype(dtype)
                except Exception:
                    pass
    return chunk


def load_csv_standard(uploaded_file) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Standard CSV loading for smaller files"""
    try:
        df = pd.read_csv(uploaded_file, low_memory=False)
        return df, None
    except Exception as e:
        return None, str(e)


def load_csv_parallel(uploaded_file, chunk_size: int, file_size_mb: float, optimize_func=None) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Parallel CSV loading for large files using chunked reading"""
    try:
        uploaded_file.seek(0)

        # First pass: read a sample to infer dtypes
        sample_df = pd.read_csv(uploaded_file, nrows=1000)
        dtypes = sample_df.dtypes.to_dict()

        # Reset file pointer
        uploaded_file.seek(0)

        # Read in chunks and process in parallel
        n_cores = max(1, mp.cpu_count() - 1)  # Leave one core free

        with mp.Pool(processes=n_cores) as pool:
            chunk_iterator = pd.read_csv(
                uploaded_file,
                chunksize=chunk_size,
                low_memory=False
            )

            process_func = partial(process_chunk, dtypes=dtypes)
            chunks = pool.map(process_func, chunk_iterator)

        df = pd.concat(chunks, ignore_index=True)

        if optimize_func is not None:
            df = optimize_func(df)

        return df, None

    except Exception as e:
        return None, f"Parallel loading error: {str(e)}"


def load_csv_dask(uploaded_file, blocksize: str = "64MB") -> Tuple[Optional[pd.DataFrame], Optional["dd.DataFrame"], Optional[str]]:
    """Load very large CSV files using Dask for out-of-core processing

    Returns preview pandas DataFrame, the dask DataFrame (for later operations), and an optional error string.
    """
    try:
        uploaded_file.seek(0)

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        try:
            ddf = dd.read_csv(tmp_path, blocksize=blocksize)
            preview_df = ddf.head(1000)
            return preview_df, ddf, None

        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except Exception as e:
        return None, None, f"Dask loading error: {str(e)}"
