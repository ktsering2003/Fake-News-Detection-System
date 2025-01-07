import streamlit as st  
import pandas as pd
import joblib

@st.cache_data
def load_data():
    return pd.read_csv("data/WELFake_Dataset.csv")
    
@st.cache_resource
def load_model():
    model = joblib.load("models/fake_news_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")  # Basically Makes sure, this matches the saved filename
    return model, vectorizer


# Load dataset and model:
df = load_data()
model, vectorizer = load_model()

# This is for streamlit UI:
st.title("Fake News Detection System")
st.title("This application allows you to explore the fake news dataset and test the model!")

# The Overview of the data set:
st.subheader("Dataset Overview:")
st.write(f"Shape of the dataset: {df.shape}")
st.write(df.head())

# User Input:
st.subheader("Check a News Article:")
user_input = st.text_area("Enter a news article or headline:")

if user_input:
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)
    confidence = model.predict_proba(user_vec).max()
    
    if prediction[0] == 1:
        st.error(f"This is likely FAKE news! Confidence: {confidence:.2f}")
    else:
        st.success(f"This is likely REAL news! Confidence: {confidence:.2f}")

        




