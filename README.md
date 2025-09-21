
# 🎬 IMDB Movie Review Sentiment Analysis

##  Problem Statement  
Classify user reviews from the IMDB dataset as either **positive** or **negative**, enabling automated sentiment detection for movie feedback.

##  Project Overview  
An end-to-end deep learning pipeline using a **GRU RNN** architecture to analyze text sentiment, integrated into a **Streamlit web app** and deployed on **Streamlit Cloud**.

##  Approach  
- **Feature Engineering**  
  - Tokenize and pad text sequences  
  - Convert categorical inputs to numerical format  
  - Standardize data for stable training  
- **Modeling**  
  - GRU RNN with Embedding layer  
  - Dropout for regularization  
  - Optimized using binary crossentropy loss  
- **Serialization**  
  - Save model weights (`.keras`) and preprocessing objects (`.pkl`) for reuse  
- **Web Integration**  
  - Streamlit app for real-time sentiment prediction  
  - Clean UI with text input and result display  
- **Deployment**  
  - Hosted on Streamlit Cloud for public access  

##  Project Structure  
```
imdb-sentiment/
├── app.py                               # Streamlit frontend
├── imdb_sentiment_analysis.keras        # Trained RNN model
├── requirements.txt                     # Dependencies
└── README.md                            # Project overview
```

## 🌐 Live Demo  
[Streamlit App Link](https://imdb-movie-review-sentiment-analysis-ayfreingpzdwjfiniyjy2g.streamlit.app/) 

## 🛠 Tech Stack  
- Python, Pandas, NumPy  
- TensorFlow, Keras  
- Streamlit  

## 📬 Contact  
For questions or collaboration, connect via [LinkedIn](https://www.linkedin.com/in/kavanakpadaki/) 
