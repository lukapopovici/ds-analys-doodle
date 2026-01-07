import streamlit as st
from utils.data_handler import DataHandler
from utils.visualizer import Visualizer
from utils.analyzer import Analyzer
from pages import overview, analysis, visualization, upload

st.set_page_config(
    page_title="DS Doodle Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    # Sidebar navigation
    st.sidebar.title("DS Doodle Pro")
    st.sidebar.markdown("---")
    
    page = st.sidebar.radio(
        "Navigation",
        ["Overview", "Upload Data", "Analysis", "Visualization"],
        label_visibility="collapsed"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("Tip: Upload your own CSV file or use the sample data to get started!")
    
    # Route to appropriate page
    if page == "Overview":
        overview.render()
    elif page == "Upload Data":
        upload.render()
    elif page == "Analysis":
        analysis.render()
    elif page == "Visualization":
        visualization.render()

if __name__ == "__main__":
    main()