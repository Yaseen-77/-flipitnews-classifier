import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import nltk
from nltk.corpus import stopwords
from tensorflow import keras

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


@st.cache_resource
def load_model():
    model = keras.models.load_model("rnn_model.keras")
    with open("rnn_assets.pkl", "rb") as f:
        assets = pickle.load(f)
    return model, assets


model, assets = load_model()
word_index = assets["word_index"]
classes = assets["classes"]
max_len = assets["max_len"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    return [w for w in tokens if w not in stop_words and len(w) > 1]


def to_sequence(text):
    ids = [word_index.get(w, 1) for w in clean_text(text)]
    return keras.utils.pad_sequences([ids], maxlen=max_len, padding="pre", truncating="post")


def predict(text):
    return model.predict(to_sequence(text), verbose=0)[0]


examples = {
    "Sports": "The team won the final match with a late goal and the coach praised the players after the league victory.",
    "Business": "The company reported higher profits this quarter as sales grew and shares rose in the stock market.",
    "Technology": "The new software update lets mobile users run apps faster using a more powerful chip.",
    "Politics": "The government announced a new budget and the minister said parliament will vote on the policy next week.",
    "Entertainment": "The film won three awards and the lead actor thanked the director at the music and film festival.",
}

st.set_page_config(page_title="FlipItNews Classifier")
st.title("FlipItNews: News Category Classifier")
st.write("Paste a news article and the RNN (LSTM) model will predict its category.")

choice = st.selectbox("Try an example", ["Write my own"] + list(examples))
text = st.text_area("News article", value=examples.get(choice, ""), height=200)

if st.button("Classify"):
    if len(text.split()) < 5:
        st.warning("Please enter a longer article (at least 5 words).")
    else:
        probs = predict(text)
        best = int(probs.argmax())
        st.success(f"Category: {classes[best]}  ({probs[best] * 100:.1f}% confident)")
        st.bar_chart(pd.DataFrame({"Probability": probs}, index=classes))