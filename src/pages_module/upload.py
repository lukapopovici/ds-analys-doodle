import streamlit as st
from utils.data_handler import DataHandler

def render():
    """Render the upload page"""
    st.title("Upload Your Data")
    st.markdown("Load your own CSV file or use our sample dataset to get started.")
    
    data_handler = DataHandler()
    
    # Create tabs for different upload options
    tab1, tab2 = st.tabs(["Upload CSV", "Use Sample Data"])
    
    with tab1:
        st.markdown("### Upload Your CSV File")
        st.markdown("Upload a CSV file to begin your analysis. The file should have headers in the first row.")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with headers"
        )
        
        if uploaded_file is not None:
            with st.spinner('Loading data...'):
                df, error = data_handler.load_from_csv(uploaded_file)
                
                if error:
                    st.error(f"Error loading file: {error}")
                else:
                    st.session_state.data = df
                    st.success(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns!")
                    
                    # Show data preview
                    st.markdown("#### Data Preview")
                    st.dataframe(df.head(20), width='stretch')
                    
                    # Show data info
                    info = data_handler.get_data_info(df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Rows", f"{info['rows']:,}")
                    with col2:
                        st.metric("Columns", info['columns'])
                    with col3:
                        st.metric("Numeric Columns", info['numeric_columns'])
                    with col4:
                        st.metric("Categorical Columns", info['categorical_columns'])
                    
                    # Column information
                    st.markdown("#### Column Information")
                    col_types = data_handler.get_column_types(df)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**Numeric Columns**")
                        if col_types['numeric']:
                            for col in col_types['numeric']:
                                st.markdown(f"- {col}")
                        else:
                            st.markdown("_None_")
                    
                    with col2:
                        st.markdown("**Categorical Columns**")
                        if col_types['categorical']:
                            for col in col_types['categorical']:
                                st.markdown(f"- {col}")
                        else:
                            st.markdown("_None_")
                    
                    with col3:
                        st.markdown("**DateTime Columns**")
                        if col_types['datetime']:
                            for col in col_types['datetime']:
                                st.markdown(f"- {col}")
                        else:
                            st.markdown("_None_")
    
    with tab2:
        st.markdown("### Use Sample Dataset")
        st.markdown("""
        Load a sample dataset to explore the application's features. This dataset contains:
        - **100 rows** of synthetic sales data
        - **6 columns**: date, sales, customers, category, region, profit
        - **Multiple data types**: dates, numeric values, and categories
        """)
        
        if st.button("Load Sample Data", type="primary"):
            with st.spinner('Creating sample dataset...'):
                df = data_handler.create_sample_data()
                st.session_state.data = df
                st.success("Sample data loaded successfully!")
                st.rerun()
        
        # Show preview if sample data is loaded
        if st.session_state.data is not None:
            st.markdown("#### Current Data Preview")
            st.dataframe(st.session_state.data.head(10), width='stretch')
    
    st.markdown("---")
    
    # Data management section
    if st.session_state.data is not None:
        st.markdown("### Data Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Clear Data", type="secondary"):
                st.session_state.data = None
                st.success("Data cleared successfully!")
                st.rerun()
        
        with col2:
            csv = data_handler.export_to_csv(st.session_state.data)
            st.download_button(
                label="Download Data as CSV",
                data=csv,
                file_name="data_export.csv",
                mime="text/csv",
                type="primary"
            )
    
    

    st.markdown("---")
