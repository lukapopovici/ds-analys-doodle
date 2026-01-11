"""
Visualization module with Strategy Pattern for flexible rendering optimization.

Design Pattern: Strategy Pattern
- Allows runtime selection of rendering strategies (parallel, sequential, downsampled)
- Easy to add new optimization strategies without modifying existing code
- Separates optimization logic from visualization logic
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from functools import partial
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from abc import ABC, abstractmethod
from enum import Enum
from .debug_decorators import log_call, log_exceptions, timeit

# Optional dependencies
try:
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error
    _HAS_SKLEARN = True
except Exception:
    LabelEncoder = None
    LinearRegression = None
    r2_score = None
    mean_squared_error = None
    _HAS_SKLEARN = False

try:
    from scipy.stats import chi2_contingency
    _HAS_SCIPY = True
except Exception:
    chi2_contingency = None
    _HAS_SCIPY = False


class DownsampleMethod(Enum):
    """Enumeration of downsampling strategies"""
    RANDOM = "random"
    SYSTEMATIC = "systematic"
    NONE = "none"


# ============================================================================
# STRATEGY PATTERN: Base Strategy Interface
# ============================================================================

class RenderingStrategy(ABC):
    """
    Abstract base class for rendering strategies.
    Each strategy implements different optimization approaches.
    """
    
    @abstractmethod
    def should_optimize(self, df, **kwargs):
        """Determine if this strategy should be applied"""
        pass
    
    @abstractmethod
    def prepare_data(self, df, **kwargs):
        """Prepare data according to strategy"""
        pass
    
    @abstractmethod
    def get_title_suffix(self, original_len, processed_len):
        """Generate informative title suffix"""
        pass


class SequentialStrategy(RenderingStrategy):
    """No optimization - render all data sequentially"""
    
    def should_optimize(self, df, **kwargs):
        return False
    
    def prepare_data(self, df, **kwargs):
        return df, {}
    
    def get_title_suffix(self, original_len, processed_len):
        return ""


class DownsampleStrategy(RenderingStrategy):
    """Downsample data for faster rendering"""
    
    def __init__(self, max_points=50000, method=DownsampleMethod.RANDOM, threshold=10000):
        self.max_points = max_points
        self.method = method
        self.threshold = threshold
    
    def should_optimize(self, df, **kwargs):
        return len(df) > self.threshold
    
    def prepare_data(self, df, **kwargs):
        if len(df) <= self.max_points:
            return df, {'downsampled': False}
        
        if self.method == DownsampleMethod.RANDOM:
            result = df.sample(n=self.max_points, random_state=42)
        elif self.method == DownsampleMethod.SYSTEMATIC:
            step = len(df) // self.max_points
            result = df.iloc[::step]
        else:
            result = df
        
        return result, {'downsampled': True, 'original_len': len(df)}
    
    def get_title_suffix(self, original_len, processed_len):
        if original_len > processed_len:
            return f" (showing {processed_len:,} of {original_len:,} points)"
        return ""


class ParallelStrategy(RenderingStrategy):
    """Use parallel processing for computations"""
    
    def __init__(self, max_workers=None, threshold=10000):
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.threshold = threshold
    
    def should_optimize(self, df, **kwargs):
        return len(df) >= self.threshold or kwargs.get('force_parallel', False)
    
    def prepare_data(self, df, **kwargs):
        # Parallel strategy doesn't transform data, just signals to use parallel execution
        return df, {'use_parallel': True, 'max_workers': self.max_workers}
    
    def get_title_suffix(self, original_len, processed_len):
        return ""


# ============================================================================
# STRATEGY CONTEXT: Manages strategy selection and execution
# ============================================================================

class StrategyContext:
    """
    Context class that uses strategies to optimize rendering.
    Handles strategy selection and coordination.
    """
    
    def __init__(self, strategies=None):
        """
        Initialize with a list of strategies (applied in order)
        
        Args:
            strategies: List of RenderingStrategy instances
        """
        self.strategies = strategies or []
    
    def add_strategy(self, strategy):
        """Add a new strategy to the context"""
        self.strategies.append(strategy)
    
    def apply_strategies(self, df, **kwargs):
        """
        Apply all applicable strategies and return optimized data
        
        Returns:
            tuple: (processed_df, metadata_dict, title_suffix)
        """
        processed_df = df
        metadata = {}
        title_parts = []
        original_len = len(df)
        
        for strategy in self.strategies:
            if strategy.should_optimize(processed_df, **kwargs):
                processed_df, strategy_meta = strategy.prepare_data(processed_df, **kwargs)
                metadata.update(strategy_meta)
                
                suffix = strategy.get_title_suffix(original_len, len(processed_df))
                if suffix:
                    title_parts.append(suffix)
        
        title_suffix = "".join(title_parts)
        return processed_df, metadata, title_suffix


# ============================================================================
# COMPUTATION STRATEGIES: For parallel computations
# ============================================================================

class ComputationStrategy(ABC):
    """Base class for computation strategies (correlation, regression, etc.)"""
    
    @abstractmethod
    def compute(self, data, **kwargs):
        """Execute the computation"""
        pass


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


# ============================================================================
# MAIN VISUALIZER: Uses strategies for optimized rendering
# ============================================================================

class Visualizer:
    """
    Main visualization class using Strategy Pattern.
    
    Design Rationale:
    - Strategy Pattern allows flexible, runtime selection of optimization strategies
    - Easy to add new strategies (e.g., GPU acceleration, adaptive sampling)
    - Separates concerns: visualization logic vs optimization logic
    - Strategies can be combined (downsample + parallel)
    """
    
    def __init__(self, 
                 enable_downsampling=True,
                 enable_parallel=True,
                 max_workers=None,
                 parallel_threshold=10000,
                 downsample_threshold=10000):
        """
        Initialize visualizer with strategy configuration
        
        Args:
            enable_downsampling: Enable automatic downsampling
            enable_parallel: Enable parallel processing
            max_workers: Number of parallel workers
            parallel_threshold: Min rows for parallel processing
            downsample_threshold: Min rows for downsampling
        """
        self.color_schemes = {
            'Plotly': px.colors.qualitative.Plotly,
            'Viridis': px.colors.sequential.Viridis,
            'Blues': px.colors.sequential.Blues,
            'Reds': px.colors.sequential.Reds,
            'Pastel': px.colors.qualitative.Pastel
        }
        
        # Initialize strategies
        self.max_workers = max_workers or max(1, mp.cpu_count() - 1)
        self.parallel_threshold = parallel_threshold
        self.downsample_threshold = downsample_threshold
        
        # Create strategy context for different chart types
        self.strategies = {
            'scatter': self._create_scatter_strategies(enable_downsampling),
            'line': self._create_line_strategies(enable_downsampling),
            'histogram': self._create_histogram_strategies(enable_downsampling),
            '3d_scatter': self._create_3d_strategies(enable_downsampling),
            'computation': self._create_computation_strategies(enable_parallel)
        }
        
        # Computation strategies
        self.correlation_strategy = (ParallelCorrelation(self.max_workers) 
                                     if enable_parallel 
                                     else SequentialCorrelation())
        self.cramers_strategy = ParallelCramersV(self.max_workers) if enable_parallel else None
    
    def _create_scatter_strategies(self, enable):
        """Create strategies for scatter plots"""
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=50000,
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold
            ))
        return StrategyContext(strategies)
    
    def _create_line_strategies(self, enable):
        """Create strategies for line charts"""
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=50000,
                method=DownsampleMethod.SYSTEMATIC,  # Preserve trends
                threshold=self.downsample_threshold
            ))
        return StrategyContext(strategies)
    
    def _create_histogram_strategies(self, enable):
        """Create strategies for histograms"""
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=100000,  # Histograms can handle more data
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold * 2
            ))
        return StrategyContext(strategies)
    
    def _create_3d_strategies(self, enable):
        """Create strategies for 3D plots"""
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=20000,  # 3D is more resource-intensive
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold // 2
            ))
        return StrategyContext(strategies)
    
    def _create_computation_strategies(self, enable):
        """Create strategies for computational operations"""
        strategies = []
        if enable:
            strategies.append(ParallelStrategy(
                max_workers=self.max_workers,
                threshold=self.parallel_threshold
            ))
        return StrategyContext(strategies)
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_scatter_plot(self, df, x_col, y_col, color_col=None, size_col=None, 
                           title="Scatter Plot"):
        """Create scatter plot with automatic optimization"""
        plot_df, metadata, title_suffix = self.strategies['scatter'].apply_strategies(df)
        
        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=title + title_suffix, hover_data=plot_df.columns)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_line_chart(self, df, x_col, y_col, color_col=None, title="Line Chart"):
        """Create line chart with automatic optimization"""
        plot_df, metadata, title_suffix = self.strategies['line'].apply_strategies(df)
        
        fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, 
                      title=title + title_suffix, markers=True)
        fig.update_layout(
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_histogram(self, df, column, nbins=30, title="Histogram"):
        """Create histogram with automatic optimization"""
        plot_df, metadata, title_suffix = self.strategies['histogram'].apply_strategies(df)
        
        fig = px.histogram(plot_df, x=column, nbins=nbins, title=title + title_suffix)
        fig.update_layout(
            template='plotly_white',
            showlegend=False
        )
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_3d_scatter(self, df, x_col, y_col, z_col, color_col=None, 
                         title="3D Scatter Plot"):
        """Create 3D scatter with automatic optimization"""
        plot_df, metadata, title_suffix = self.strategies['3d_scatter'].apply_strategies(df)
        
        fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=color_col, 
                           title=title + title_suffix)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_bar_chart(self, df, x_col, y_col, color_col=None, title="Bar Chart"):
        """Create bar chart"""
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                    title=title, text_auto=True)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_box_plot(self, df, y_col, x_col=None, title="Box Plot"):
        """Create box plot"""
        fig = px.box(df, y=y_col, x=x_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_pie_chart(self, df, names_col, values_col=None, title="Pie Chart"):
        """Create pie chart"""
        if values_col is None:
            data = df[names_col].value_counts().reset_index()
            data.columns = [names_col, 'count']
            fig = px.pie(data, names=names_col, values='count', title=title)
        else:
            fig = px.pie(df, names=names_col, values=values_col, title=title)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_area_chart(self, df, x_col, y_col, color_col=None, title="Area Chart"):
        """Create area chart with automatic optimization"""
        plot_df, metadata, title_suffix = self.strategies['line'].apply_strategies(df)
        
        fig = px.area(plot_df, x=x_col, y=y_col, color=color_col, title=title + title_suffix)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @log_call(level="DEBUG")
    def create_violin_plot(self, df, y_col, x_col=None, title="Violin Plot"):
        """Create violin plot"""
        fig = px.violin(df, y=y_col, x=x_col, title=title, box=True)
        fig.update_layout(template='plotly_white')
        return fig
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def create_heatmap(self, df, title="Correlation Heatmap", columns=None):
        """Create correlation heatmap with strategy-based optimization"""
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for create_heatmap")
        
        if columns:
            df_subset = df[columns].copy()
        else:
            df_subset = df.select_dtypes(include=[np.number]).copy()
        
        # Encode categorical and datetime columns
        df_encoded = df_subset.copy()
        le = LabelEncoder()
        for col in df_encoded.columns:
            if pd.api.types.is_datetime64_any_dtype(df_encoded[col]):
                df_encoded[col] = df_encoded[col].astype('int64') / 10**9
            elif df_encoded[col].dtype == 'object':
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
        # Use strategy to compute correlation
        _, metadata, _ = self.strategies['computation'].apply_strategies(
            df_encoded, 
            force_parallel=(len(df_encoded.columns) > 5)
        )
        
        if metadata.get('use_parallel', False):
            corr = self.correlation_strategy.compute(df_encoded)
        else:
            corr = df_encoded.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=700,
            height=700
        )
        return fig
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def create_categorical_heatmap(self, df, categorical_cols, 
                                   title="Categorical Correlation Heatmap"):
        """Create categorical correlation heatmap with parallel processing"""
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is required for create_categorical_heatmap")
        
        _, metadata, _ = self.strategies['computation'].apply_strategies(
            df, 
            force_parallel=(len(categorical_cols) > 5)
        )
        
        if metadata.get('use_parallel', False) and self.cramers_strategy:
            corr_matrix = self.cramers_strategy.compute(df, categorical_cols)
        else:
            # Sequential fallback
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                min_dim = min(confusion_matrix.shape) - 1
                return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            n = len(categorical_cols)
            corr_matrix = np.zeros((n, n))
            for i, col1 in enumerate(categorical_cols):
                for j, col2 in enumerate(categorical_cols):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        corr_matrix[i, j] = cramers_v(df[col1], df[col2])
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=categorical_cols,
            y=categorical_cols,
            colorscale='Viridis',
            text=corr_matrix.round(2),
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title="Cramér's V")
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            width=700,
            height=700
        )
        return fig
    
    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def perform_linear_regression(self, df, x_cols, y_col):
        """Perform linear regression with strategy-based optimization"""
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for perform_linear_regression")
        
        X = df[x_cols].copy()
        y = df[y_col].copy()
        
        # Encode data
        le = LabelEncoder()
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype('int64') / 10**9
            elif X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        
        X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
        y = pd.to_numeric(y, errors='coerce').fillna(y.mean())
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # Apply downsampling strategy for visualization
        plot_df = pd.DataFrame({'actual': y, 'predicted': predictions})
        vis_df, metadata, title_suffix = self.strategies['scatter'].apply_strategies(plot_df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vis_df['actual'],
            y=vis_df['predicted'],
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6)
        ))
        
        min_val = min(y.min(), predictions.min())
        max_val = max(y.max(), predictions.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Linear Regression: Actual vs Predicted{title_suffix}<br>R² = {r2:.4f}, RMSE = {rmse:.4f}',
            xaxis_title=f'Actual {y_col}',
            yaxis_title=f'Predicted {y_col}',
            template='plotly_white'
        )
        
        results = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'coefficients': dict(zip(x_cols, model.coef_)),
            'intercept': model.intercept_
        }
        
        return model, predictions, results, fig
    
    def create_residual_plot(self, df, y_col, predictions):
        """Create residual plot with downsampling"""
        residuals = df[y_col] - predictions
        plot_df = pd.DataFrame({'predicted': predictions, 'residuals': residuals})
        vis_df, metadata, title_suffix = self.strategies['scatter'].apply_strategies(plot_df)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vis_df['predicted'],
            y=vis_df['residuals'],
            mode='markers',
            marker=dict(size=8, opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title=f'Residual Plot{title_suffix}',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template='plotly_white'
        )
        return fig
    
    def create_feature_importance_plot(self, feature_names, coefficients, 
                                      title="Feature Importance"):
        """Create feature importance plot"""
        if isinstance(coefficients, dict):
            coefficients = np.array([coefficients[name] for name in feature_names])
        
        importance = np.abs(coefficients)
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        sorted_coefficients = coefficients[indices]
        
        colors = ['green' if c > 0 else 'red' for c in sorted_coefficients]
        
        fig = go.Figure(data=[
            go.Bar(
                x=sorted_importance,
                y=sorted_features,
                orientation='h',
                marker=dict(color=colors),
                text=[f'{c:.4f}' for c in sorted_coefficients],
                textposition='auto'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title='Absolute Coefficient Value',
            yaxis_title='Features',
            template='plotly_white',
            height=max(400, len(feature_names) * 30)
        )
        return fig