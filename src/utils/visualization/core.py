"""Core Visualizer class that composes strategies and computation helpers."""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from .strategies import DownsampleMethod, DownsampleStrategy, ParallelStrategy
from .context import StrategyContext
from .computations import SequentialCorrelation, ParallelCorrelation, ParallelCramersV
from ..debug_decorators import log_call, log_exceptions, timeit

# Optional dependencies used in some methods
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

# SciPy (used for fallback Cramér's V computation as well)
try:
    from scipy.stats import chi2_contingency
    _HAS_SCIPY = True
except Exception:
    chi2_contingency = None
    _HAS_SCIPY = False


class Visualizer:
    """
    Main visualization class using Strategy Pattern.
    """

    def __init__(self,
                 enable_downsampling=True,
                 enable_parallel=True,
                 max_workers=None,
                 parallel_threshold=10000,
                 downsample_threshold=10000):
        self.color_schemes = {
            'Plotly': px.colors.qualitative.Plotly,
            'Viridis': px.colors.sequential.Viridis,
            'Blues': px.colors.sequential.Blues,
            'Reds': px.colors.sequential.Reds,
            'Pastel': px.colors.qualitative.Pastel
        }

        self.max_workers = max_workers or max(1, __import__('multiprocessing').cpu_count() - 1)
        self.parallel_threshold = parallel_threshold
        self.downsample_threshold = downsample_threshold

        self.strategies = {
            'scatter': self._create_scatter_strategies(enable_downsampling),
            'line': self._create_line_strategies(enable_downsampling),
            'histogram': self._create_histogram_strategies(enable_downsampling),
            '3d_scatter': self._create_3d_strategies(enable_downsampling),
            'computation': self._create_computation_strategies(enable_parallel)
        }

        self.correlation_strategy = (ParallelCorrelation(self.max_workers)
                                     if enable_parallel
                                     else SequentialCorrelation())
        self.cramers_strategy = ParallelCramersV(self.max_workers) if enable_parallel else None

    def _create_scatter_strategies(self, enable):
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=50000,
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold
            ))
        return StrategyContext(strategies)

    def _create_line_strategies(self, enable):
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=50000,
                method=DownsampleMethod.SYSTEMATIC,  # Preserve trends
                threshold=self.downsample_threshold
            ))
        return StrategyContext(strategies)

    def _create_histogram_strategies(self, enable):
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=100000,  # Histograms can handle more data
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold * 2
            ))
        return StrategyContext(strategies)

    def _create_3d_strategies(self, enable):
        strategies = []
        if enable:
            strategies.append(DownsampleStrategy(
                max_points=20000,  # 3D is more resource-intensive
                method=DownsampleMethod.RANDOM,
                threshold=self.downsample_threshold // 2
            ))
        return StrategyContext(strategies)

    def _create_computation_strategies(self, enable):
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
        plot_df, metadata, title_suffix = self.strategies['scatter'].apply_strategies(df)

        fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, size=size_col,
                         title=title + title_suffix, hover_data=plot_df.columns)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @log_call(level="DEBUG")
    def create_line_chart(self, df, x_col, y_col, color_col=None, title="Line Chart"):
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
        plot_df, metadata, title_suffix = self.strategies['3d_scatter'].apply_strategies(df)

        fig = px.scatter_3d(plot_df, x=x_col, y=y_col, z=z_col, color=color_col,
                            title=title + title_suffix)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @log_call(level="DEBUG")
    def create_bar_chart(self, df, x_col, y_col, color_col=None, title="Bar Chart"):
        fig = px.bar(df, x=x_col, y=y_col, color=color_col,
                     title=title, text_auto=True)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @log_call(level="DEBUG")
    def create_box_plot(self, df, y_col, x_col=None, title="Box Plot"):
        fig = px.box(df, y=y_col, x=x_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @log_call(level="DEBUG")
    def create_pie_chart(self, df, names_col, values_col=None, title="Pie Chart"):
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
        plot_df, metadata, title_suffix = self.strategies['line'].apply_strategies(df)

        fig = px.area(plot_df, x=x_col, y=y_col, color=color_col, title=title + title_suffix)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @log_call(level="DEBUG")
    def create_violin_plot(self, df, y_col, x_col=None, title="Violin Plot"):
        fig = px.violin(df, y=y_col, x=x_col, title=title, box=True)
        fig.update_layout(template='plotly_white')
        return fig

    @log_exceptions()
    @timeit(level="INFO")
    @log_call(level="DEBUG")
    def create_heatmap(self, df, title="Correlation Heatmap", columns=None):
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for create_heatmap")

        if columns:
            df_subset = df[columns].copy()
        else:
            df_subset = df.select_dtypes(include=[np.number]).copy()

        df_encoded = df_subset.copy()
        le = LabelEncoder()
        for col in df_encoded.columns:
            if pd.api.types.is_datetime64_any_dtype(df_encoded[col]):
                df_encoded[col] = df_encoded[col].astype('int64') / 10**9
            elif df_encoded[col].dtype == 'object':
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

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
        if not _HAS_SCIPY:
            raise RuntimeError("scipy is required for create_categorical_heatmap")

        _, metadata, _ = self.strategies['computation'].apply_strategies(
            df,
            force_parallel=(len(categorical_cols) > 5)
        )

        if metadata.get('use_parallel', False) and self.cramers_strategy:
            corr_matrix = self.cramers_strategy.compute(df, categorical_cols)
        else:
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
        if not _HAS_SKLEARN:
            raise RuntimeError("scikit-learn is required for perform_linear_regression")

        X = df[x_cols].copy()
        y = df[y_col].copy()

        le = LabelEncoder()
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype('int64') / 10**9
            elif X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))

        X = X.apply(pd.to_numeric, errors='coerce').fillna(X.mean())
        y = pd.to_numeric(y, errors='coerce').fillna(y.mean())

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

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