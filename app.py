import streamlit as st
import joblib
import numpy as np
from openai import OpenAI

st.set_page_config(page_title="MindTrack AI", layout="centered")

# LOGO + TITLE
col1, col2 = st.columns([1, 5])

with col1:
   st.image("logo.png", width=70)

with col2:
    st.title("MindTrack AI")

st.write("AI-powered Mental Health Predictor + Chatbot")


# تحميل الموديل

model = joblib.load("rf_model.pkl")


# OPENAI AI SETUP

client = OpenAI(api_key="YOUR_API_KEY")

def bot_response(msg):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a supportive mental health assistant. Be kind, short, and helpful."},
            {"role": "user", "content": msg}
        ]
    )
    return response.choices[0].message.content


# INPUT

st.header("Enter Your Data")

sleep_time = st.slider("Sleep Hours", 0, 12, 7)
screen_time = st.slider("Screen Time (hours)", 0, 12, 5)
messages = st.slider("Messages per day", 0, 200, 50)
calls = st.slider("Call duration (min)", 0, 300, 30)


# PREDICTION

if st.button("🔍 Predict Stress Level"):

    input_data = np.array([[sleep_time, screen_time, messages, calls]])

    prediction = model.predict(input_data)[0]

    st.subheader("Result")

    if prediction == 2:
        st.error("High Stress 🔴")
    elif prediction == 1:
        st.warning("Moderate Stress 🟡")
    else:
        st.success("Low Stress 🟢")

    st.write("💡 AI Insight:")
    st.write("Based on your behavior patterns, your mental health risk was estimated using Random Forest model.")


# CHATBOT

st.divider()
st.header("AI Chatbot")

if "chat" not in st.session_state:
    st.session_state.chat = []

for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.write(m["text"])

user = st.chat_input("Ask about mental health...")

if user:
    st.session_state.chat.append({"role": "user", "text": user})

    reply = bot_response(user)

    st.session_state.chat.append({"role": "assistant", "text": reply})

    st.rerun()
