
# 🎬 IMDB Movie Review Sentiment Analysis

##  Problem Statement  
Classify user reviews from the IMDB dataset as either **positive** or **negative**, enabling automated sentiment detection for movie feedback.

##  Project Overview  
An end-to-end deep learning pipeline using a **Simple RNN** architecture to analyze text sentiment, integrated into a **Streamlit web app** and deployed on **Streamlit Cloud**.

##  Approach  
- **Feature Engineering**  
  - Tokenize and pad text sequences  
  - Convert categorical inputs to numerical format  
  - Standardize data for stable training  
- **Modeling**  
  - Simple RNN with Embedding layer  
  - Dropout for regularization  
  - Optimized using binary crossentropy loss  
- **Serialization**  
  - Save model weights (`.h5`) and preprocessing objects (`.pkl`) for reuse  
- **Web Integration**  
  - Streamlit app for real-time sentiment prediction  
  - Clean UI with text input and result display  
- **Deployment**  
  - Hosted on Streamlit Cloud for public access  

##  Project Structure  
```
imdb-sentiment/
├── app.py                 # Streamlit frontend
├── model.h5               # Trained RNN model
├── tokenizer.pkl          # Tokenizer object
├── requirements.txt       # Dependencies
└── README.md              # Project overview
```

## 🌐 Live Demo  
[Streamlit App Link](https://your-app-url.streamlit.app) 

## 🛠 Tech Stack  
- Python, Pandas, NumPy  
- TensorFlow, Keras  
- Streamlit  

## 📬 Contact  
For questions or collaboration, connect via [LinkedIn](https://www.linkedin.com/in/your-profile) 
