import streamlit as st
from .chart_constants import CHART_TYPES


def render():
    """Page to toggle which chart types are available in the Visualization page."""
    st.title("Chart Settings")

    st.markdown(
        "Choose which chart types should appear in the Visualization page. "
        "You can also limit the number of visible charts (0 = show all enabled)."
    )

    # Initialize session defaults if not present
    if 'enabled_charts' not in st.session_state:
        st.session_state['enabled_charts'] = CHART_TYPES.copy()
    if 'max_visible_charts' not in st.session_state:
        st.session_state['max_visible_charts'] = 0

    enabled = st.session_state.get('enabled_charts', CHART_TYPES.copy())

    st.markdown("### Enable / disable chart types")
    selected = st.multiselect("Enabled charts (order preserved)", CHART_TYPES, default=enabled)

    st.markdown("---")
    st.markdown("### Display limit")
    max_visible = st.slider("Max visible charts (0 = no limit)", 0, len(CHART_TYPES), value=st.session_state.get('max_visible_charts', 0))

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Save settings"):
            st.session_state['enabled_charts'] = selected
            st.session_state['max_visible_charts'] = max_visible
            st.success("Chart settings saved")
            st.rerun()
    with col2:
        if st.button("Reset to defaults"):
            st.session_state['enabled_charts'] = CHART_TYPES.copy()
            st.session_state['max_visible_charts'] = 0
            st.success("Reset to defaults")
            st.rerun()

    st.markdown("---")

    if st.button("Go to Visualization Page"):
        st.switch_page("pages_module/visualization.py")
