import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# Set page config
st.set_page_config(page_title="AI Data Insights", layout="wide")

# Load OpenAI API key from Streamlit secrets
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Title
st.title("📊 Data Insights with AI 🔍")

# Upload CSV
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Display Data
    st.subheader("🔍 Data Preview")
    st.dataframe(df)

    # Summary Statistics
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    # Plotting
    st.subheader("📉 Histogram")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    if numeric_cols:
        col_to_plot = st.selectbox("Choose a numeric column to plot", numeric_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col_to_plot].dropna(), bins=20, color='lightblue', edgecolor='black')
        ax.set_title(f"Histogram of {col_to_plot}")
        st.pyplot(fig)
    else:
        st.warning("No numeric columns available to plot.")

    # AI Insights
    st.subheader("🧠 AI Insights")

    prompt = f"""
    Analyze this data table and give meaningful insights, patterns, or anomalies in bullet points.

    {df.head(10).to_string(index=False)}
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional data analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        ai_reply = response.choices[0].message.content
        st.markdown(ai_reply)
    except Exception as e:
        st.error("Something went wrong while generating insights.")
        st.exception(e)
