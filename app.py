import streamlit as st
from transformers import pipeline
from huggingface_hub import login
from dotenv import load_dotenv
import os
import torch

# Load .env (optional)
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# If you have a token, login (not required for public models)
if hf_token:
    login(token=hf_token)

# Set page config
st.set_page_config(page_title="Sentiment Analysis", layout="wide")

st.title("🧠 Sentiment Analysis")
st.write("Type a sentence in English, and the model will classify its sentiment.")

# Text input from user
text = st.text_area("✍️ Your sentence:", height=100)

# Button
if st.button("🔍 Analyze Sentiment"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        # Select device (GPU if available)
        device = 0 if torch.cuda.is_available() else -1

        # Load pipeline
        classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

        # Run prediction
        result = classifier(text)[0]
        label = result["label"]       # "POSITIVE" or "NEGATIVE"
        score = float(result["score"])  # confidence

        # Choose color
        if label == "POSITIVE":
            color = "green"
            emoji = "😊"
        else:
            color = "red"
            emoji = "😞"

        # Show result
        st.markdown(
            f"### {emoji} Sentiment: <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True
        )
        st.markdown(f"**Confidence:** `{score:.3f}`")

        # Friendly explanation
        if label == "POSITIVE":
            st.success("The model thinks this sentence expresses a positive sentiment.")
        else:
            st.error("The model thinks this sentence expresses a negative sentiment.")
