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
        st.title("Basic Streamlit Interface")
        st.write("This is a simple Streamlit app using a class.")

        name = st.text_input("Enter your name")

        if st.button("Submit"):
            if name:
                st.success(f"Hello, {name}!")
            else:
                st.warning("Please enter your name.")

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
        page_title="My First Streamlit App",
        layout="centered"
    )

    app = DataScienceApp()
    option = app.render_sidebar()
    app.render_main(option)


if __name__ == "__main__":
    main()
