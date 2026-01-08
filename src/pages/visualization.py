import streamlit as st
import pandas as pd
from utils.visualizer import Visualizer
from utils.data_handler import DataHandler
from utils.terminal import terminal_toggle_ui

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
            "Categorical Heatmap",
            "3D Scatter Plot",
            "Linear Regression",
        ]
    )
    
    st.sidebar.markdown("---")

    # Terminal toggle (compact, does NOT render messages)
    try:
        terminal_toggle_ui(st.sidebar)
    except Exception:
        # Fail gracefully if Streamlit isn't available for the helper
        pass

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
        st.markdown("""
        Create a correlation heatmap to see relationships between variables.
        You can select specific columns (both numeric and categorical) or use all numeric columns.
        """)
        
        use_selected = st.checkbox("Select specific columns", value=False)
        
        if use_selected:
            selected_cols = st.multiselect(
                "Select columns to include",
                all_cols,
                default=numeric_cols[:min(5, len(numeric_cols))] if numeric_cols else []
            )
            
            if len(selected_cols) < 2:
                st.warning("Please select at least 2 columns for correlation analysis.")
            else:
                chart_title = st.text_input("Chart Title", "Correlation Heatmap")
                
                if st.button("Generate Chart", type="primary"):
                    with st.spinner('Creating visualization...'):
                        fig = visualizer.create_heatmap(df, chart_title, selected_cols)
                        st.plotly_chart(fig, use_container_width=True)
        else:
            if len(numeric_cols) < 2:
                st.warning("Need at least 2 numeric columns for correlation heatmap.")
            else:
                chart_title = st.text_input("Chart Title", "Correlation Heatmap")
                
                if st.button("Generate Chart", type="primary"):
                    with st.spinner('Creating visualization...'):
                        fig = visualizer.create_heatmap(df, chart_title)
                        st.plotly_chart(fig, use_container_width=True)
    
    elif chart_type == "Categorical Heatmap":
        st.markdown("""
        Create a heatmap showing relationships between categorical variables using Cramér's V.
        This measures the association between categorical variables (0 = no association, 1 = perfect association).
        """)
        
        if len(categorical_cols) < 2:
            st.warning("Need at least 2 categorical columns for categorical correlation analysis.")
        else:
            selected_cats = st.multiselect(
                "Select categorical columns",
                categorical_cols,
                default=categorical_cols[:min(5, len(categorical_cols))]
            )
            
            if len(selected_cats) < 2:
                st.warning("Please select at least 2 categorical columns.")
            else:
                chart_title = st.text_input("Chart Title", "Categorical Variable Relationships")
                
                if st.button("Generate Chart", type="primary"):
                    with st.spinner('Creating visualization...'):
                        fig = visualizer.create_categorical_heatmap(df, selected_cats, chart_title)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.info("""
                        **Interpreting Cramér's V:**
                        - 0.0 - 0.1: Negligible association
                        - 0.1 - 0.3: Weak association
                        - 0.3 - 0.5: Moderate association
                        - 0.5+: Strong association
                        """)
    
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
    
    elif chart_type == "Linear Regression":
        st.markdown("""
        Perform linear regression to predict one variable based on others.
        This helps discover relationships and make predictions.
        """)
        
        if len(all_cols) < 2:
            st.warning("Need at least 2 columns for regression analysis.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Target Variable (what to predict)**")
                y_col = st.selectbox("Target (Y)", numeric_cols, key="reg_y")
            
            with col2:
                st.markdown("**Feature Variables (predictors)**")
                available_features = [col for col in all_cols if col != y_col]
                x_cols = st.multiselect(
                    "Features (X)",
                    available_features,
                    default=available_features[:min(3, len(available_features))]
                )
            
            if len(x_cols) == 0:
                st.warning("Please select at least one feature variable.")
            else:
                if st.button("Run Linear Regression", type="primary"):
                    with st.spinner('Training model...'):
                        try:
                            model, predictions, results, fig = visualizer.perform_linear_regression(
                                df, x_cols, y_col
                            )
                            
                            # Display results
                            st.success("Regression analysis complete!")
                            
                            # Metrics
                            st.markdown("### Model Performance")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("R² Score", f"{results['r2_score']:.4f}")
                            with col2:
                                st.metric("RMSE", f"{results['rmse']:.4f}")
                            with col3:
                                st.metric("MSE", f"{results['mse']:.4f}")
                            
                            st.markdown("""
                            **R² Score Interpretation:**
                            - 1.0: Perfect prediction
                            - 0.7+: Strong model
                            - 0.4-0.7: Moderate model
                            - <0.4: Weak model
                            """)
                            
                            # Actual vs Predicted plot
                            st.markdown("### Actual vs Predicted Values")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Residual plot
                            st.markdown("### Residual Plot")
                            residual_fig = visualizer.create_residual_plot(df, y_col, predictions)
                            st.plotly_chart(residual_fig, use_container_width=True)
                            
                            st.info("""
                            **Residual Plot:** Points should be randomly scattered around zero.
                            Patterns indicate the model might be missing important relationships.
                            """)
                            
                            # Feature importance
                            st.markdown("### Feature Importance")
                            importance_fig = visualizer.create_feature_importance_plot(
                                x_cols,
                                results['coefficients'],
                                "Feature Coefficients"
                            )
                            st.plotly_chart(importance_fig, use_container_width=True)
                            
                            # Coefficients table
                            st.markdown("### Model Coefficients")
                            coef_df = pd.DataFrame({
                                'Feature': x_cols,
                                'Coefficient': results['coefficients']
                            })
                            coef_df['Absolute Impact'] = coef_df['Coefficient'].abs()
                            coef_df = coef_df.sort_values('Absolute Impact', ascending=False)
                            st.dataframe(coef_df, use_container_width=True, hide_index=True)
                            
                            st.markdown(f"**Intercept:** {results['intercept']:.4f}")
                            
                            # Model equation
                            st.markdown("### Model Equation")
                            equation = f"{y_col} = {results['intercept']:.4f}"
                            for feature, coef in results['coefficients'].items():
                                sign = "+" if coef >= 0 else ""
                                equation += f" {sign} {coef:.4f} × {feature}"
                            st.code(equation, language=None)
                            
                        except Exception as e:
                            st.error(f"Error running regression: {str(e)}")
                            st.info("Make sure your data doesn't have too many missing values.")
    

    st.markdown("---")
