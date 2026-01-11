import pandas as pd
from typing import List, Tuple
import multiprocessing as mp
from functools import partial


def filter_data(df: pd.DataFrame, column: str, operator: str, value) -> pd.DataFrame:
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


def _apply_filters_to_chunk(chunk: pd.DataFrame, filters: List[Tuple[str, str, any]]) -> pd.DataFrame:
    """Apply multiple filters to a chunk"""
    result = chunk.copy()
    for column, operator, value in filters:
        result = filter_data(result, column, operator, value)
    return result


def filter_data_parallel(df: pd.DataFrame, filters: List[Tuple[str, str, any]]) -> pd.DataFrame:
    """Apply multiple filters in parallel for large datasets"""
    if len(filters) == 0:
        return df

    # For single filter, use standard method
    if len(filters) == 1:
        return filter_data(df, *filters[0])

    # Split dataframe into chunks
    n_cores = max(1, mp.cpu_count() - 1)
    chunk_size = len(df) // n_cores
    chunks = [df.iloc[i:i + chunk_size] for i in range(0, len(df), chunk_size)]

    # Apply filters to each chunk in parallel
    with mp.Pool(processes=n_cores) as pool:
        filter_func = partial(_apply_filters_to_chunk, filters=filters)
        filtered_chunks = pool.map(filter_func, chunks)

    return pd.concat(filtered_chunks, ignore_index=True)


def sort_data(df: pd.DataFrame, column: str, ascending: bool = True) -> pd.DataFrame:
    """Sort dataframe by column"""
    return df.sort_values(by=column, ascending=ascending)


def export_to_csv(df: pd.DataFrame, chunk_export: bool = False) -> str:
    """Export dataframe to CSV string"""
    if chunk_export and len(df) > 100000:
        return df.to_csv(index=False, chunksize=50000)
    return df.to_csv(index=False)
