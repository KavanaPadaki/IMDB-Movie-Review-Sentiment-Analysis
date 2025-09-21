import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import numpy as np
import json

# Load the pre-trained model
model = load_model('imdb_sentiment_analysis.keras')

# Load the word index
# You need to make sure you have the word_index available, perhaps save it earlier
# For now, we will assume it is available or loaded from a file
# If you saved it as a JSON file:
# with open('word_index.json', 'r') as f:
#     word_index = json.load(f)
# If you generated it from the imdb dataset loading:
from tensorflow.keras.datasets import imdb
max_features = 10000 # Make sure this matches the value used in training
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
word_index = imdb.get_word_index()


max_len = 500 # Make sure this matches the value used in training

def preprocess_text(text):
  words = text.lower().split()
  encoded_review = [word_index.get(word, 2) + 3 for word in words]
  padded_review = sequence.pad_sequences([encoded_review], maxlen=max_len)
  return padded_review

def predict_sentiment(review):
  preprocessed_review = preprocess_text(review)
  prediction = model.predict(preprocessed_review)[0][0]
  if prediction > 0.6:
    sentiment = "Positive"
  elif prediction < 0.4:
      sentiment = "Negative"
  else:
      sentiment = "Neutral"
  return sentiment, prediction

st.title("IMDB Movie Review Sentiment Analysis")

st.write("Enter a movie review to classify its sentiment.")

review_input = st.text_area("Movie Review")

if st.button("Analyze Sentiment"):
  if review_input:
    sentiment, score = predict_sentiment(review_input)
    st.write(f"Sentiment: {sentiment}")
    st.write(f"Score: {score:.4f}")
  else:
    st.write("Please enter a review.")
