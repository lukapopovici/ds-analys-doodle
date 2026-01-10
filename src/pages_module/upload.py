import streamlit as st
from utils.data_handler import DataHandler

def render():
    """Render the upload page with parallel processing support"""
    st.title("Upload Your Data")
    st.markdown("Load your own CSV file or use our sample dataset to get started.")
    
    data_handler = DataHandler()
    
    # Create tabs for different upload options
    tab1, tab2, tab3 = st.tabs(["Upload CSV", "Use Sample Data", "Advanced Settings"])
    
    with tab1:
        st.markdown("### Upload Your CSV File")
        st.markdown("Upload a CSV file to begin your analysis. Large files (>100MB) will automatically use parallel processing.")
        
        # Advanced options
        with st.expander("Upload Options"):
            col1, col2 = st.columns(2)
            
            with col1:
                use_parallel = st.selectbox(
                    "Processing Mode",
                    options=["Auto-detect", "Standard", "Parallel", "Dask (Very Large Files)"],
                    help="Auto-detect uses parallel processing for files >100MB"
                )
            
            with col2:
                chunk_size = st.number_input(
                    "Chunk Size (rows)",
                    min_value=1000,
                    max_value=500000,
                    value=50000,
                    step=10000,
                    help="Number of rows to process at once"
                )
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload a CSV file with headers"
        )
        
        if uploaded_file is not None:
            # Show file info
            file_size_mb = uploaded_file.size / (1024 ** 2)
            st.info(f"File size: {file_size_mb:.2f} MB")
            
            # Determine processing mode
            if use_parallel == "Auto-detect":
                parallel_mode = None
                use_dask = False
            elif use_parallel == "Standard":
                parallel_mode = False
                use_dask = False
            elif use_parallel == "Parallel":
                parallel_mode = True
                use_dask = False
            else:  # Dask
                parallel_mode = False
                use_dask = True
            
            # Load button
            if st.button("Load Data", type="primary"):
                with st.spinner('Loading data...'):
                    # Add progress indicator for large files
                    if file_size_mb > 50:
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        status_text.text("Processing large file...")
                    
                    if use_dask:
                        df, error = data_handler.load_from_csv_dask(uploaded_file)
                        if error is None:
                            st.session_state.use_dask = True
                            st.session_state.dask_data = data_handler._dask_data
                    else:
                        df, error = data_handler.load_from_csv(
                            uploaded_file, 
                            use_parallel=parallel_mode,
                            chunk_size=chunk_size
                        )
                        if error is None:
                            st.session_state.use_dask = False
                    
                    if file_size_mb > 50:
                        progress_bar.progress(100)
                        status_text.text("Loading complete!")
                    
                    if error:
                        st.error(f"Error loading file: {error}")
                    else:
                        st.session_state.data = df
                        
                        # Success message with processing info
                        if use_dask:
                            st.success(f" Successfully loaded data using Dask! Showing preview of {len(df):,} rows.")
                            st.info("Full dataset is loaded in memory-efficient mode. Some operations will process data in chunks.")
                        elif parallel_mode or (parallel_mode is None and file_size_mb > 100):
                            st.success(f"Successfully loaded {len(df):,} rows using parallel processing!")
                        else:
                            st.success(f" Successfully loaded {len(df):,} rows!")
                        
                        # Show data preview
                        st.markdown("#### Data Preview")
                        st.dataframe(df.head(20), use_container_width=True)
                        
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
                        
                        # Memory usage info
                        st.markdown("#### Memory Usage")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Memory Usage", f"{info['memory']:.2f} MB")
                        with col2:
                            st.metric("Missing Values", f"{info['missing_values']:,}")
                        
                        # Memory optimization suggestion
                        memory_report = data_handler.get_memory_usage_report(df)
                        if memory_report['optimized']:
                            st.warning("This dataset can be optimized to use less memory. Use the 'Optimize Memory' button below.")
                            if st.button("Optimize Memory Usage"):
                                with st.spinner("Optimizing..."):
                                    optimized_df = data_handler._optimize_dtypes(df)
                                    st.session_state.data = optimized_df
                                    new_info = data_handler.get_data_info(optimized_df)
                                    reduction = ((info['memory'] - new_info['memory']) / info['memory']) * 100
                                    st.success(f"Memory usage reduced by {reduction:.1f}% ({info['memory']:.2f} MB → {new_info['memory']:.2f} MB)")
                                    st.rerun()
                        
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
        
    
        elif st.session_state.data is not None:
            st.info(" Data already loaded. Upload a new file to replace it.")
            st.markdown("#### Current Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
    
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
                st.session_state.use_dask = False
                st.success("Sample data loaded successfully!")
                st.rerun()
        
        # naaaah
        if st.session_state.data is not None:
            st.markdown("#### Current Data Preview")
            st.dataframe(st.session_state.data.head(10), use_container_width=True)
    
    with tab3:
        st.markdown("### Advanced Settings")
        st.markdown("Configure advanced options for data processing.")
        
        st.markdown("#### Parallel Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input(
                "CPU Cores to Use",
                min_value=1,
                max_value=16,
                value=4,
                help="Number of CPU cores for parallel processing"
            )
        
        with col2:
            st.number_input(
                "Memory Limit (MB)",
                min_value=100,
                max_value=32000,
                value=2000,
                help="Maximum memory to use for processing"
            )
        
        st.markdown("#### File Size Thresholds")
        
        st.slider(
            "Auto-parallel threshold (MB)",
            min_value=50,
            max_value=500,
            value=100,
            help="Files larger than this will automatically use parallel processing"
        )
        
        st.markdown("#### Performance Tips")
        st.info("""
        **For best performance:**
        - Files < 100 MB: Use standard mode
        - Files 100 MB - 1 GB: Use parallel mode
        - Files > 1 GB: Use Dask mode
        - Enable memory optimization for datasets with repeated values
        """)
    
    st.markdown("---")
    
    # Data management section
    if st.session_state.data is not None:
        st.markdown("### Data Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Data", type="secondary"):
                st.session_state.data = None
                st.session_state.use_dask = False
                if 'dask_data' in st.session_state:
                    del st.session_state.dask_data
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
        
        with col3:
            # Show detailed memory report
            if st.button("View Memory Report"):
                memory_report = data_handler.get_memory_usage_report(st.session_state.data)
                st.markdown("#### Memory Usage by Column")
                
                # Sort columns by MEM usage
                sorted_cols = sorted(
                    memory_report['per_column'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                for col, mem in sorted_cols[:10]:  # Top 10
                    st.text(f"{col}: {mem:.2f} MB")
    
    st.markdown("---")