import streamlit as st
from utils.data_handler import DataHandler

def render():
    """Render the overview page"""
    st.markdown('<h1 class="main-header">DS Doodle Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Your Complete Data Science Dashboard")
    
    # Welcome message
    st.markdown("""
    Welcome to **DS Doodle Pro** - a powerful, interactive data science application built with Streamlit!
    
    #### Features
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Data Management**
        - Upload CSV files
        - Use sample datasets
        - Filter and sort data
        - Export processed data
        
        **Advanced Analysis**
        - Descriptive statistics
        - Correlation analysis
        - Outlier detection
        - Normality tests
        """)
    
    with col2:
        st.markdown("""
        **Rich Visualizations**
        - Interactive Plotly charts
        - Multiple chart types
        - Customizable styling
        - 3D visualizations
        
        **Smart Insights**
        - Group statistics
        - Missing value reports
        - Distribution analysis
        - Trend identification
        """)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### Quick Start Guide")
    
    with st.expander("1. Load Your Data", expanded=True):
        st.markdown("""
        Navigate to the **Upload Data** page to:
        - Upload your own CSV file, or
        - Load the sample dataset to explore features
        """)
    
    with st.expander("2. Analyze Your Data"):
        st.markdown("""
        Go to the **Analysis** page to:
        - View descriptive statistics
        - Check correlations
        - Detect outliers
        - Generate insights
        """)
    
    with st.expander("3. Create Visualizations"):
        st.markdown("""
        Visit the **Visualization** page to:
        - Create interactive charts
        - Customize plot types
        - Export visualizations
        - Explore relationships
        """)
    
    st.markdown("---")
    
    # Current data status
    st.markdown("### Current Data Status")
    
    if st.session_state.data is not None:
        data_handler = DataHandler()
        info = data_handler.get_data_info(st.session_state.data)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{info['rows']:,}")
        with col2:
            st.metric("Total Columns", f"{info['columns']}")
        with col3:
            st.metric("Memory Usage", f"{info['memory']:.2f} MB")
        with col4:
            st.metric("Missing Values", f"{info['missing_values']:,}")
        
        st.success("Data is loaded and ready for analysis!")
        
        # Show quick preview
        st.markdown("#### Quick Preview")
        st.dataframe(st.session_state.data.head(10), use_container_width=True)
        
    else:
        st.info("No data loaded yet. Head to the **Upload Data** page to get started!")
        
        if st.button("Load Sample Data Now", type="primary"):
            data_handler = DataHandler()
            st.session_state.data = data_handler.create_sample_data()
            st.rerun()
    
    st.markdown("---")
    
    # Tips section
    st.markdown("### Pro Tips")
    st.markdown("""
    - **Start with exploration**: Use the Analysis page to understand your data before creating visualizations
    - **Check for missing values**: Always review the missing values report before analysis
    - **Use filters**: Apply filters in the Analysis page to focus on specific subsets of your data
    - **Export results**: You can download processed data and visualizations for use in reports
    - **Experiment**: Try different chart types to find the best way to tell your data story
    """)
    
    st.markdown("---")
    st.markdown("Built with Streamlit | DS Doodle Pro")