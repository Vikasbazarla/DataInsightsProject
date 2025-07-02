import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="CSV Data Insights", layout="wide")
st.title("📊 CSV Data Analysis Dashboard")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load CSV
    df = pd.read_csv(uploaded_file)
    st.success("File uploaded successfully!")

    # Show basic details
    st.subheader("🔍 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("🧮 Summary Statistics")
    st.write(df.describe())

    st.subheader("📈 Null Values Heatmap")
    fig1, ax1 = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", ax=ax1)
    st.pyplot(fig1)

    st.subheader("📊 Correlation Matrix")
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        fig2, ax2 = plt.subplots()
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)
    else:
        st.warning("No numeric columns found for correlation heatmap.")

    st.subheader("📌 Column-wise Distributions")
    for col in numeric_df.columns:
        fig3, ax3 = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax3)
        ax3.set_title(f'Distribution of {col}')
        st.pyplot(fig3)

else:
    st.info("Please upload a CSV file to get started.")

