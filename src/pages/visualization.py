import streamlit as st
import pandas as pd
from utils.visualizer import Visualizer
from utils.data_handler import DataHandler

def render():
    """Render the visualization page"""
    st.title("Data Visualization")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload data first!")
        if st.button("Go to Upload Page"):
            st.switch_page("pages/upload.py")
        return
    
    df = st.session_state.data
    visualizer = Visualizer()
    data_handler = DataHandler()
    
    # Get column types
    col_types = data_handler.get_column_types(df)
    numeric_cols = col_types['numeric']
    categorical_cols = col_types['categorical']
    all_cols = list(df.columns)
    
    # Sidebar for chart configuration
    st.sidebar.markdown("### Chart Configuration")
    
    chart_type = st.sidebar.selectbox(
        "Select Chart Type",
        [
            "Line Chart",
            "Scatter Plot",
            "Bar Chart",
            "Histogram",
            "Box Plot",
            "Pie Chart",
            "Area Chart",
            "Violin Plot",
            "Correlation Heatmap",
            "3D Scatter Plot"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Main visualization area
    st.markdown(f"### {chart_type}")
    
    # Chart-specific configurations
    if chart_type == "Line Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", all_cols, key="line_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="line_y")
        
        with col2:
            color_col = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="line_color")
            chart_title = st.text_input("Chart Title", "Line Chart")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                color = None if color_col == "None" else color_col
                fig = visualizer.create_line_chart(df, x_col, y_col, color, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Scatter Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
        
        with col2:
            color_col = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="scatter_color")
            size_col = st.selectbox("Size By (optional)", ["None"] + numeric_cols, key="scatter_size")
        
        chart_title = st.text_input("Chart Title", "Scatter Plot")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                color = None if color_col == "None" else color_col
                size = None if size_col == "None" else size_col
                fig = visualizer.create_scatter_plot(df, x_col, y_col, color, size, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Bar Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", all_cols, key="bar_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="bar_y")
        
        with col2:
            color_col = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="bar_color")
            chart_title = st.text_input("Chart Title", "Bar Chart")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                color = None if color_col == "None" else color_col
                fig = visualizer.create_bar_chart(df, x_col, y_col, color, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Histogram":
        col1, col2 = st.columns(2)
        
        with col1:
            column = st.selectbox("Select Column", numeric_cols, key="hist_col")
            nbins = st.slider("Number of Bins", 10, 100, 30)
        
        with col2:
            chart_title = st.text_input("Chart Title", f"Distribution of {column if 'column' in locals() else 'Variable'}")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                fig = visualizer.create_histogram(df, column, nbins, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Box Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            y_col = st.selectbox("Y-axis (Value)", numeric_cols, key="box_y")
            x_col = st.selectbox("X-axis (Category, optional)", ["None"] + categorical_cols, key="box_x")
        
        with col2:
            chart_title = st.text_input("Chart Title", "Box Plot")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                x = None if x_col == "None" else x_col
                fig = visualizer.create_box_plot(df, y_col, x, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Pie Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            names_col = st.selectbox("Categories", categorical_cols, key="pie_names")
            values_col = st.selectbox("Values (optional)", ["None"] + numeric_cols, key="pie_values")
        
        with col2:
            chart_title = st.text_input("Chart Title", "Pie Chart")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                values = None if values_col == "None" else values_col
                fig = visualizer.create_pie_chart(df, names_col, values, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Area Chart":
        col1, col2 = st.columns(2)
        
        with col1:
            x_col = st.selectbox("X-axis", all_cols, key="area_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="area_y")
        
        with col2:
            color_col = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="area_color")
            chart_title = st.text_input("Chart Title", "Area Chart")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                color = None if color_col == "None" else color_col
                fig = visualizer.create_area_chart(df, x_col, y_col, color, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Violin Plot":
        col1, col2 = st.columns(2)
        
        with col1:
            y_col = st.selectbox("Y-axis (Value)", numeric_cols, key="violin_y")
            x_col = st.selectbox("X-axis (Category, optional)", ["None"] + categorical_cols, key="violin_x")
        
        with col2:
            chart_title = st.text_input("Chart Title", "Violin Plot")
        
        if st.button("Generate Chart", type="primary"):
            with st.spinner('Creating visualization...'):
                x = None if x_col == "None" else x_col
                fig = visualizer.create_violin_plot(df, y_col, x, chart_title)
                st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Correlation Heatmap":
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation heatmap.")
        else:
            chart_title = st.text_input("Chart Title", "Correlation Heatmap")
            
            if st.button("Generate Chart", type="primary"):
                with st.spinner('Creating visualization...'):
                    fig = visualizer.create_heatmap(df, chart_title)
                    st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "3D Scatter Plot":
        if len(numeric_cols) < 3:
            st.warning("Need at least 3 numeric columns for 3D scatter plot.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                x_col = st.selectbox("X-axis", numeric_cols, key="3d_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="3d_y")
                z_col = st.selectbox("Z-axis", numeric_cols, key="3d_z")
            
            with col2:
                color_col = st.selectbox("Color By (optional)", ["None"] + categorical_cols, key="3d_color")
                chart_title = st.text_input("Chart Title", "3D Scatter Plot")
            
            if st.button("Generate Chart", type="primary"):
                with st.spinner('Creating visualization...'):
                    color = None if color_col == "None" else color_col
                    fig = visualizer.create_3d_scatter(df, x_col, y_col, z_col, color, chart_title)
                    st.plotly_chart(fig, use_container_width=True)
    
 
    
    st.markdown("---")
