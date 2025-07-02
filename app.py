import streamlit as st
import pandas as pd
import openai

# Set the page title
st.set_page_config(page_title="AI Data Insights", layout="centered")

st.title("📊 AI Data Insights from CSV")

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Load API key from secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("🔍 Preview of Your Data")
    st.write(df.head())

    # This is the missing part:
    if st.button("Generate AI Insights"):
        with st.spinner("Analyzing your data with AI..."):
            prompt = f"Provide 5 interesting insights from this dataset:\n{df.head(15).to_csv(index=False)}"

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            result = response['choices'][0]['message']['content']
            st.subheader("🧠 AI Insights")
            st.write(result)
