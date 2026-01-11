import streamlit as st
import pandas as pd
from utils.analyzer import Analyzer
from utils.data_handler import DataHandler

def render():
    """Render the analysis page"""
    st.title("Data Analysis")
    
    if st.session_state.data is None:
        st.warning("No data loaded. Please upload data first!")
        if st.button("Go to Upload Page"):
            st.switch_page("pages_module/upload.py")
        return
    
    df = st.session_state.data
    analyzer = Analyzer()
    data_handler = DataHandler()
    
    # Create tabs for different analysis types
    tabs = st.tabs([
        "Overview",
        "Descriptive Stats",
        "Correlations",
        "Outliers",
        "Value Counts",
        "Missing Values",
        "Group Analysis"
    ])
    
    # Overview Tab
    with tabs[0]:
        st.markdown("### Dataset Overview")
        
        info = data_handler.get_data_info(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", f"{info['rows']:,}")
            st.metric("Total Columns", info['columns'])
        with col2:
            st.metric("Numeric Columns", info['numeric_columns'])
            st.metric("Categorical Columns", info['categorical_columns'])
        with col3:
            st.metric("Memory Usage", f"{info['memory']:.2f} MB")
            st.metric("Missing Values", f"{info['missing_values']:,}")
        
        st.markdown("#### Data Preview")
        
        # Add filtering options
        with st.expander("Filter Data"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_col = st.selectbox("Select Column", ["None"] + list(df.columns))
            
            if filter_col != "None":
                with col2:
                    if pd.api.types.is_numeric_dtype(df[filter_col]):
                        operator = st.selectbox("Operator", ["equals", "greater than", "less than"])
                        with col3:
                            filter_value = st.number_input("Value", value=float(df[filter_col].mean()))
                    else:
                        operator = st.selectbox("Operator", ["equals", "contains"])
                        with col3:
                            filter_value = st.text_input("Value")
                
                if st.button("Apply Filter"):
                    df = data_handler.filter_data(df, filter_col, operator, filter_value)
                    st.success(f"Filter applied! Showing {len(df)} rows.")
        
        # Sorting options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            sort_col = st.selectbox("Sort by Column", df.columns)
        with col2:
            sort_order = st.radio("Order", ["Ascending", "Descending"])
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Sort"):
                df = data_handler.sort_data(df, sort_col, sort_order == "Ascending")
        
        # Show dataframe
        st.dataframe(df, width='stretch', height=400)
        
        # Download filtered data
        csv = data_handler.export_to_csv(df)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name="filtered_data.csv",
            mime="text/csv"
        )
    
    # Descriptive Statistics Tab
    with tabs[1]:
        st.markdown("### Descriptive Statistics")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            selected_column = st.selectbox(
                "Select Column for Analysis",
                df.columns,
                key="desc_stats_col"
            )
            
            if st.button("Generate Statistics", type="primary"):
                with col2:
                    with st.spinner('Calculating statistics...'):
                        stats_df = analyzer.get_descriptive_stats(df, selected_column)
                        st.dataframe(stats_df, width='stretch', hide_index=True)
                        
                        # Additional visualizations for numeric columns
                        if pd.api.types.is_numeric_dtype(df[selected_column]):
                            st.markdown("#### Distribution")
                            st.bar_chart(df[selected_column].value_counts().head(20))
    
    # Correlations Tab
    with tabs[2]:
        st.markdown("### Correlation Analysis")
        st.markdown("Analyze relationships between numeric variables.")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("Need at least 2 numeric columns for correlation analysis.")
        else:
            corr_matrix = analyzer.get_correlation_matrix(df)
            
            st.markdown("#### Correlation Matrix")
            st.dataframe(corr_matrix.style.background_gradient(cmap='coolwarm', axis=None), 
                        width='stretch')
            
            # Find strongest correlations
            st.markdown("#### Strongest Correlations")
            
            # Get upper triangle of correlation matrix
            corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_pairs.append({
                        'Variable 1': corr_matrix.columns[i],
                        'Variable 2': corr_matrix.columns[j],
                        'Correlation': corr_matrix.iloc[i, j]
                    })
            
            corr_df = pd.DataFrame(corr_pairs)
            corr_df['Abs Correlation'] = corr_df['Correlation'].abs()
            corr_df = corr_df.sort_values('Abs Correlation', ascending=False).head(10)
            
            st.dataframe(
                corr_df[['Variable 1', 'Variable 2', 'Correlation']],
                width='stretch',
                hide_index=True
            )
    
    # Outliers Tab
    with tabs[3]:
        st.markdown("### Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            st.warning("No numeric columns available for outlier detection.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                outlier_col = st.selectbox("Select Column", numeric_cols, key="outlier_col")
            
            with col2:
                method = st.selectbox("Detection Method", 
                                     ["IQR (Interquartile Range)", "Z-Score"],
                                     help="IQR: Q1 - 1.5*IQR and Q3 + 1.5*IQR\nZ-Score: |z| > 3")
            
            if st.button("Detect Outliers", type="primary"):
                method_key = 'iqr' if 'IQR' in method else 'zscore'
                outliers, message = analyzer.detect_outliers(df, outlier_col, method_key)
                
                st.info(message)
                
                if outliers is not None and len(outliers) > 0:
                    st.markdown("#### Detected Outliers")
                    st.dataframe(outliers, width='stretch')
                    
                    # Visualize
                    st.markdown("#### Distribution with Outliers Highlighted")
                    st.scatter_chart(df[[outlier_col]])
                else:
                    st.success("No outliers detected!")
    
    # Value Counts Tab
    with tabs[4]:
        st.markdown("### Value Counts")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            value_col = st.selectbox("Select Column", df.columns, key="value_counts_col")
        
        with col2:
            top_n = st.slider("Show Top N Values", 5, 50, 10)
        
        if st.button("Generate Value Counts", type="primary"):
            counts_df = analyzer.get_value_counts(df, value_col, top_n)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Value Counts Table")
                st.dataframe(counts_df, width='stretch', hide_index=True)
            
            with col2:
                st.markdown("#### Visualization")
                st.bar_chart(counts_df.set_index(value_col))
    
    # Missing Values Tab
    with tabs[5]:
        st.markdown("### Missing Values Analysis")
        
        missing_report = analyzer.get_missing_values_report(df)
        
        if len(missing_report) == 0:
            st.success("No missing values found in the dataset!")
        else:
            st.warning(f"Found missing values in {len(missing_report)} columns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Missing Values Report")
                st.dataframe(missing_report, width='stretch', hide_index=True)
            
            with col2:
                st.markdown("#### Missing Values Visualization")
                st.bar_chart(missing_report.set_index('Column')['Percentage'])
    
    # Group Analysis Tab
    with tabs[6]:
        st.markdown("### Group Statistics")
        st.markdown("Analyze numeric variables grouped by categories.")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not categorical_cols or not numeric_cols:
            st.warning("Need both categorical and numeric columns for group analysis.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                group_col = st.selectbox("Group By", categorical_cols, key="group_col")
            
            with col2:
                value_col = st.selectbox("Analyze", numeric_cols, key="group_value_col")
            
            if st.button("Generate Group Statistics", type="primary"):
                grouped_stats, error = analyzer.get_group_statistics(df, group_col, value_col)
                
                if error:
                    st.error(f"Error: {error}")
                else:
                    st.markdown("#### Group Statistics Table")
                    st.dataframe(grouped_stats, width='stretch', hide_index=True)
                    
                    st.markdown("#### Mean by Group")
                    st.bar_chart(grouped_stats.set_index(group_col)['Mean'])