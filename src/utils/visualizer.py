import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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
    
    def create_heatmap(self, df, title="Correlation Heatmap"):
        """Create correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title=title,
            template='plotly_white'
        )
        return fig
    
    def create_pie_chart(self, df, names_col, values_col=None, title="Pie Chart"):
        """Create pie chart"""
        if values_col is None:
            # count occurrences
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