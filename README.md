# DS_AI-Echo--Your-Smartest-Conversational-Partner

📄 Project Description

AI Echo is an interactive web application that performs sentiment analysis on user reviews. Using Natural Language Processing (NLP) and a Logistic Regression model, it classifies reviews into Positive, Neutral, or Negative sentiments.

The project also includes an EDA Dashboard and Model Performance section to help you explore the dataset and evaluate the classifier.

Features

✅ Text Preprocessing: Converts reviews to lowercase, removes stopwords (but keeps negation words like 'not', 'never'), and applies lemmatization.
✅ Sentiment Labeling: Maps numeric ratings to sentiment: 4–5 → Positive, 3 → Neutral, 1–2 → Negative.
✅ Data Balancing: Uses random downsampling to balance classes.
✅ Machine Learning Pipeline: TF-IDF vectorization, Logistic Regression model with class balancing, model evaluation (classification report & confusion matrix).
✅ Streamlit App: 
  - Prediction: Enter any text review and predict its sentiment.
  - EDA Dashboard: View sentiment distribution, ratings breakdown, and word clouds.
  - Model Performance: See classification metrics and confusion matrix.
    
Tech Stack

- Python
- Pandas, Seaborn, Matplotlib
- NLTK for NLP tasks
- Scikit-learn for machine learning
- WordCloud for word cloud visualization
- Streamlit for the interactive web app

  Future Improvements

- Add more advanced models (Random Forest, XGBoost, LSTM)
- Improve text preprocessing with bigrams/trigrams
- Add more visualizations (word frequency, sentiment trends)
- Deploy on the web (Streamlit Community Cloud, Heroku, or similar)


