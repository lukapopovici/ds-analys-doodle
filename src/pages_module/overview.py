import streamlit as st
from utils.data_handler import DataHandler

def render():
    """Render the overview page"""
    st.markdown('<h1 class="main-header">SCOPE</h1>', unsafe_allow_html=True)

    st.markdown("""
   Hey!
    """)
    
    st.markdown("---")
 
    st.markdown("### current data_Status")
    
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
        st.dataframe(st.session_state.data.head(40), width='stretch')
        
    else:
        st.info("No data loaded.")
        
        if st.button("Load Sample Data Now", type="primary"):
            data_handler = DataHandler()
            st.session_state.data = data_handler.create_sample_data()
            st.rerun()
    

    
    st.markdown("---")
