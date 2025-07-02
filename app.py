import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI

# Set Streamlit page config
st.set_page_config(page_title="AI Data Insights", layout="wide")
st.title("📊 AI-Powered Data Insights")
st.write("Upload a CSV file and let OpenAI generate insights automatically.")

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📌 Preview of Your Data")
    st.dataframe(df.head())

    # Show basic summary
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())

    # Prepare data for AI prompt
    sample_data = df.head(10).to_string()

    # Generate AI insights
    st.subheader("🤖 AI-Generated Insights")

    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    prompt = f"""You are a data analyst. Analyze the following data table and provide useful, easy-to-understand insights.
    
    Data:
    {sample_data}
    
    Return your response as bullet points."""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI data analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    ai_reply = response.choices[0].message.content
    st.markdown(ai_reply)

    # Optional: Plot if numeric columns exist
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) >= 2:
        st.subheader("📊 Data Plot")
        col1 = st.selectbox("Choose X-axis", numeric_cols)
        col2 = st.selectbox("Choose Y-axis", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)

        fig, ax = plt.subplots()
        ax.scatter(df[col1], df[col2])
        ax.set_xlabel(col1)
        ax.set_ylabel(col2)
        ax.set_title(f"{col1} vs {col2}")
        st.pyplot(fig)

