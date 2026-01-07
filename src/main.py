import streamlit as st

# Data science imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class DataScienceApp:
    def __init__(self):
        self.data = self.load_data()

    def load_data(self):
        """Create or load example data"""
        df = pd.DataFrame({
            "x": np.arange(10),
            "y": np.random.randn(10)
        })
        return df

    def render_sidebar(self):
        st.sidebar.header("Sidebar")
        return st.sidebar.selectbox(
            "Choose an option",
            ["Show Data", "Show Plot"]
        )

    def render_main(self, option):
        st.title("DS Doodle")
    


        if option == "Show Data":
            st.subheader("Data Preview")
            st.dataframe(self.data)

        elif option == "Show Plot":
            st.subheader("Simple Plot")
            fig, ax = plt.subplots()
            ax.plot(self.data["x"], self.data["y"])
            st.pyplot(fig)


def main():
    st.set_page_config(
        page_title="DS Doodle",
        layout="centered"
    )

    app = DataScienceApp()
    option = app.render_sidebar()
    app.render_main(option)


if __name__ == "__main__":
    main()
