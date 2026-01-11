import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# Optional dependency: scipy
try:
    from scipy.stats import chi2_contingency
    _HAS_SCIPY = True
except Exception:
    chi2_contingency = None
    _HAS_SCIPY = False


class ComputationStrategy:
    """Base class for computation strategies (correlation, regression, etc.)"""

    def compute(self, data, **kwargs):
        """Execute the computation"""
        raise NotImplementedError()


class SequentialCorrelation(ComputationStrategy):
    """Sequential correlation computation"""

    def compute(self, df_encoded, **kwargs):
        return df_encoded.corr()


class ParallelCorrelation(ComputationStrategy):
    """Parallel correlation computation using threading"""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def compute(self, df_encoded, **kwargs):
        cols = df_encoded.columns.tolist()
        n_cols = len(cols)
        corr_matrix = np.zeros((n_cols, n_cols))

        def calc_corr_row(i):
            row = np.zeros(n_cols)
            col_i = df_encoded.iloc[:, i].values
            for j in range(n_cols):
                if i == j:
                    row[j] = 1.0
                else:
                    col_j = df_encoded.iloc[:, j].values
                    row[j] = np.corrcoef(col_i, col_j)[0, 1]
            return i, row

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(calc_corr_row, range(n_cols))
            for i, row in results:
                corr_matrix[i, :] = row

        return pd.DataFrame(corr_matrix, columns=cols, index=cols)


class ParallelCramersV(ComputationStrategy):
    """Parallel Cramér's V computation"""

    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def compute(self, df, categorical_cols, **kwargs):
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is required for Cramér's V computation")

        def cramers_v(x, y):
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0

        def calc_cell(args):
            i, j, col1, col2 = args
            if i == j:
                return (i, j, 1.0)
            return (i, j, cramers_v(df[col1], df[col2]))

        n = len(categorical_cols)
        corr_matrix = np.zeros((n, n))

        tasks = [(i, j, categorical_cols[i], categorical_cols[j])
                 for i in range(n) for j in range(i, n)]

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = executor.map(calc_cell, tasks)
            for i, j, val in results:
                corr_matrix[i, j] = val
                corr_matrix[j, i] = val

        return corr_matrix
