import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

class Visualizer:
    """Handle all visualization operations"""
    
    def __init__(self):
        self.color_schemes = {
            'Plotly': px.colors.qualitative.Plotly,
            'Viridis': px.colors.sequential.Viridis,
            'Blues': px.colors.sequential.Blues,
            'Reds': px.colors.sequential.Reds,
            'Pastel': px.colors.qualitative.Pastel
        }
    
    def create_line_chart(self, df, x_col, y_col, color_col=None, title="Line Chart"):
        """Create interactive line chart"""
        fig = px.line(df, x=x_col, y=y_col, color=color_col, 
                      title=title, markers=True)
        fig.update_layout(
            hovermode='x unified',
            template='plotly_white'
        )
        return fig
    
    def create_scatter_plot(self, df, x_col, y_col, color_col=None, size_col=None, title="Scatter Plot"):
        """Create interactive scatter plot"""
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col, size=size_col,
                        title=title, hover_data=df.columns)
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_bar_chart(self, df, x_col, y_col, color_col=None, title="Bar Chart"):
        """Create interactive bar chart"""
        fig = px.bar(df, x=x_col, y=y_col, color=color_col, 
                    title=title, text_auto=True)
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_histogram(self, df, column, nbins=30, title="Histogram"):
        """Create histogram"""
        fig = px.histogram(df, x=column, nbins=nbins, title=title)
        fig.update_layout(
            template='plotly_white',
            showlegend=False
        )
        return fig
    
    def create_box_plot(self, df, y_col, x_col=None, title="Box Plot"):
        """Create box plot"""
        fig = px.box(df, y=y_col, x=x_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_heatmap(self, df, title="Correlation Heatmap", columns=None):
        """Create correlation heatmap for numeric or encoded categorical columns"""
        from sklearn.preprocessing import LabelEncoder
        
        if columns:
            # Use only selected columns
            df_subset = df[columns].copy()
        else:
            # Use all numeric columns
            df_subset = df.select_dtypes(include=[np.number]).copy()
        
        # Encode categorical and datetime columns if present
        df_encoded = df_subset.copy()
        le = LabelEncoder()
        for col in df_encoded.columns:
            # Convert datetime to numeric (unix timestamp)
            if pd.api.types.is_datetime64_any_dtype(df_encoded[col]):
                df_encoded[col] = df_encoded[col].astype('int64') / 10**9
            # Encode categorical variables
            elif df_encoded[col].dtype == 'object':
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        
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
    
    def create_categorical_heatmap(self, df, categorical_cols, title="Categorical Correlation Heatmap"):
        """Create heatmap showing relationships between categorical variables using Cramér's V"""
        from scipy.stats import chi2_contingency
        
        def cramers_v(x, y):
            """Calculate Cramér's V statistic for categorical-categorical association"""
            confusion_matrix = pd.crosstab(x, y)
            chi2 = chi2_contingency(confusion_matrix)[0]
            n = confusion_matrix.sum().sum()
            min_dim = min(confusion_matrix.shape) - 1
            return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
        
        # Create correlation matrix
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
    
    def create_pie_chart(self, df, names_col, values_col=None, title="Pie Chart"):
        """Create pie chart"""
        if values_col is None:
            # Count occurrences
            data = df[names_col].value_counts().reset_index()
            data.columns = [names_col, 'count']
            fig = px.pie(data, names=names_col, values='count', title=title)
        else:
            fig = px.pie(df, names=names_col, values=values_col, title=title)
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_area_chart(self, df, x_col, y_col, color_col=None, title="Area Chart"):
        """Create area chart"""
        fig = px.area(df, x=x_col, y=y_col, color=color_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_violin_plot(self, df, y_col, x_col=None, title="Violin Plot"):
        """Create violin plot"""
        fig = px.violin(df, y=y_col, x=x_col, title=title, box=True)
        fig.update_layout(template='plotly_white')
        return fig
    
    def create_3d_scatter(self, df, x_col, y_col, z_col, color_col=None, title="3D Scatter Plot"):
        """Create 3D scatter plot"""
        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col, title=title)
        fig.update_layout(template='plotly_white')
        return fig
    
    def perform_linear_regression(self, df, x_cols, y_col):
        """
        Perform linear regression with multiple features
        Returns: model, predictions, metrics, and visualization figure
        """
        from sklearn.preprocessing import LabelEncoder
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import r2_score, mean_squared_error
        
        # Prepare data
        X = df[x_cols].copy()
        y = df[y_col].copy()
        
        # Handle different data types
        le = LabelEncoder()
        for col in X.columns:
            # Convert datetime to numeric (unix timestamp)
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].astype('int64') / 10**9  # Convert to seconds
            # Encode categorical variables
            elif X[col].dtype == 'object':
                X[col] = le.fit_transform(X[col].astype(str))
        
        # Convert to numeric and handle missing values
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(X.mean())
        y = pd.to_numeric(y, errors='coerce')
        y = y.fillna(y.mean())
        
        # Fit model
        model = LinearRegression()
        model.fit(X, y)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Calculate metrics
        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        
        # Create visualization
        fig = go.Figure()
        
        # Actual vs Predicted scatter
        fig.add_trace(go.Scatter(
            x=y,
            y=predictions,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
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
            title=f'Linear Regression: Actual vs Predicted<br>R² = {r2:.4f}, RMSE = {rmse:.4f}',
            xaxis_title=f'Actual {y_col}',
            yaxis_title=f'Predicted {y_col}',
            template='plotly_white'
        )
        
        # Prepare results
        results = {
            'r2_score': r2,
            'mse': mse,
            'rmse': rmse,
            'coefficients': dict(zip(x_cols, model.coef_)),
            'intercept': model.intercept_
        }
        
        return model, predictions, results, fig
    
    def create_residual_plot(self, df, y_col, predictions):
        """Create residual plot for regression analysis"""
        residuals = df[y_col] - predictions
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=predictions,
            y=residuals,
            mode='markers',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title='Residual Plot',
            xaxis_title='Predicted Values',
            yaxis_title='Residuals',
            template='plotly_white'
        )
        
        return fig
    
    def create_feature_importance_plot(self, feature_names, coefficients, title="Feature Importance"):
        """Create bar plot showing feature importance from regression"""
        # Convert dict to list if needed
        if isinstance(coefficients, dict):
            coefficients = np.array([coefficients[name] for name in feature_names])
        
        # Get absolute values for importance
        importance = np.abs(coefficients)
        
        # Sort by importance
        indices = np.argsort(importance)[::-1]
        sorted_features = [feature_names[i] for i in indices]
        sorted_importance = importance[indices]
        sorted_coefficients = coefficients[indices]
        
        # Create color based on positive/negative
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