# Sentiment Analysis on Movie Reviews

This project is a machine learning application that analyzes the sentiment of movie reviews and classifies them as Positive or Negative. It uses traditional NLP techniques with a logistic regression model trained on TF-IDF vectors.

The app includes a simple Flask web interface where users can enter their own review text and get instant sentiment feedback.

---

## Features

- Takes raw movie reviews as input
- Uses a trained logistic regression model
- Web-based interface using Flask
- Real-time sentiment classification
- Lightweight and fast — runs locally without GPUs

---

## How It Works

1. Preprocessed movie reviews are vectorized using TF-IDF
2. A Logistic Regression model is trained on labeled reviews (positive/negative)
3. The model predicts sentiment on new user input
4. Flask serves the UI to interact with the model

---

## Folder Structure

sentiment_analysis/
├── app.py                  # Flask web app  
├── sentiment_model.py      # ML model training script  
├── reviews.csv             # Training data (500 movie reviews)  
├── sentiment_model.pkl     # Saved logistic regression model  
├── tfidf_vectorizer.pkl    # Saved TF-IDF vectorizer  
├── templates/  
│   └── index.html          # UI template for user input  

---

## How to Run the Project Locally

Step 1: Install Requirements

Make sure Python and pip are installed.

Then open Command Prompt and run:

pip install flask scikit-learn pandas joblib

Step 2: Train the Model (if not already trained)

python sentiment_model.py

This will generate:

- sentiment_model.pkl
- tfidf_vectorizer.pkl

Step 3: Launch the Web App

python app.py

Open your browser and visit:

http://127.0.0.1:5000

Type a review like:

"The plot was amazing and the acting was brilliant."

The app will respond with:

Positive

---

## Sample Data (reviews.csv)

Contains 500 rows like:

review,sentiment  
This movie was fantastic!,positive  
I hated the plot and acting.,negative

You can extend this with real reviews from IMDb, Rotten Tomatoes, etc.

## Acknowledgements

- Scikit-learn  
- Flask  
- IMDb Dataset (Kaggle)  
