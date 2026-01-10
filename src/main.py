import streamlit as st
from utils.data_handler import DataHandler
from utils.visualizer import Visualizer
from utils.analyzer import Analyzer
from utils.terminal import terminal
from pages_module import overview, analysis, visualization, upload
from utils.css_config import list_css_configs, load_css, save_uploaded_css, CSS_DIR
st.set_page_config(
    page_title="DS Doodle Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


def apply_css_from_text(css_text: str) -> None:
    """Apply raw CSS text into the Streamlit page."""
    if css_text:
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'data_handler' not in st.session_state:
        st.session_state.data_handler = DataHandler()
    
    # Sidebar navigation
    st.sidebar.title("DS Doodle Pro")
    st.sidebar.markdown("---")

    # Theme CSS loader
    css_options = list_css_configs()
    if not css_options:
        css_options = ["default"]
    if 'css_theme' not in st.session_state or st.session_state.get('css_theme') not in css_options:
        st.session_state['css_theme'] = css_options[0]

    css_choice = st.sidebar.selectbox("Theme CSS", css_options, index=css_options.index(st.session_state['css_theme']))
    if css_choice != st.session_state['css_theme']:
        st.session_state['css_theme'] = css_choice

    # Apply selected CSS
    try:
        css_text = load_css(st.session_state['css_theme'])
        apply_css_from_text(css_text)
    except Exception as e:
        st.sidebar.error(f"Could not load CSS config: {e}")

    # Upload CSS (applied immediately; can optionally save to configs)
    uploaded_css = st.sidebar.file_uploader("Upload CSS (applied immediately)", type=['css'])
    if uploaded_css:
        uploaded_text = uploaded_css.getvalue().decode('utf-8')
        apply_css_from_text(uploaded_text)
        save_name = st.sidebar.text_input("Save uploaded as", value="", key="save_css_name")
        if save_name:
            if st.sidebar.button("Save uploaded CSS", key="save_css_btn"):
                try:
                    path = save_uploaded_css(save_name.strip(), uploaded_text)
                    st.sidebar.success(f"Saved to {path.name}")
                    st.experimental_rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to save CSS: {e}")

    if 'terminal_enabled' not in st.session_state:
        st.session_state['terminal_enabled'] = terminal.is_enabled()
        terminal.enable()
        terminal.info("Application started")

    new_terminal_val = st.sidebar.checkbox("Enable terminal logging", value=st.session_state['terminal_enabled'])
    if new_terminal_val != st.session_state['terminal_enabled']:
        st.session_state['terminal_enabled'] = new_terminal_val
        terminal.set_enabled(new_terminal_val)

    st.sidebar.write(f"Terminal: {'ENABLED' if terminal.is_enabled() else 'DISABLED'} — {terminal.message_count()} messages")
    if st.sidebar.button("Clear terminal history"):
        terminal.clear()
        st.sidebar.success("Terminal history cleared")

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