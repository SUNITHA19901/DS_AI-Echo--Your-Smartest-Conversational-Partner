

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample


nltk.download('stopwords')
nltk.download('wordnet')


DATASET_PATH = r"D:\AI ECHO\Scripts\chatgpt_style_reviews_dataset.xlsx - Sheet1.csv"

try:
    df = pd.read_csv(DATASET_PATH)

except FileNotFoundError:
    st.error(f"❌ File not found: {DATASET_PATH}")
    st.stop()


df = df[['review', 'rating']]


df.dropna(subset=['review', 'rating'], inplace=True)
df['review'] = df['review'].astype(str).str.lower()


df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df.dropna(subset=['rating'], inplace=True)


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def preprocess_text(text):
    tokens = text.split()
    cleaned = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned)

df['cleaned_review'] = df['review'].apply(preprocess_text)


def get_sentiment(rating):
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    elif rating < 3:
        return 'Negative'
    else:
        return 'Unknown'

df['sentiment'] = df['rating'].apply(get_sentiment)
df = df[df['sentiment'] != 'Unknown']


positive = df[df['sentiment'] == 'Positive']
neutral = df[df['sentiment'] == 'Neutral']
negative = df[df['sentiment'] == 'Negative']

min_len = min(len(positive), len(neutral), len(negative))

positive = resample(positive, replace=False, n_samples=min_len, random_state=42)
neutral = resample(neutral, replace=False, n_samples=min_len, random_state=42)
negative = resample(negative, replace=False, n_samples=min_len, random_state=42)

df_balanced = pd.concat([positive, neutral, negative])
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)


tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df_balanced['cleaned_review'])
y = df_balanced['sentiment']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


st.set_page_config(page_title="AI Echo", layout="wide")
st.title("🧠 AI Echo: ChatGPT-Style Review Sentiment Predictor")

menu = st.sidebar.selectbox("Menu", ["Prediction", "EDA Dashboard", "Model Performance"])

if menu == "Prediction":
    st.header("🔍 Test a Review")
    user_input = st.text_area("Enter a review:")
    if st.button("Predict Sentiment"):
        if not user_input.strip():
            st.warning("⚠️ Please enter a review first.")
        else:
            cleaned_input = preprocess_text(user_input)
            vector = tfidf.transform([cleaned_input])
            prediction = model.predict(vector)[0]
            st.success(f"✅ Predicted Sentiment: **{prediction}**")

elif menu == "EDA Dashboard":
    st.header("📊 EDA Dashboard")

    st.subheader("Sentiment Distribution (Balanced)")
    fig1 = sns.countplot(data=df_balanced, x='sentiment', palette='pastel')
    st.pyplot(fig1.figure)

    st.subheader("Rating Distribution")
    st.bar_chart(df_balanced['rating'].value_counts().sort_index())

    st.subheader("Word Clouds")
    col1, col2, col3 = st.columns(3)
    with col1:
        wc_pos = WordCloud(width=500, height=300, background_color='white').generate(
            ' '.join(df_balanced[df_balanced['sentiment'] == 'Positive']['cleaned_review']))
        st.image(wc_pos.to_array(), caption="Positive Reviews")
    with col2:
        wc_neu = WordCloud(width=500, height=300, background_color='white').generate(
            ' '.join(df_balanced[df_balanced['sentiment'] == 'Neutral']['cleaned_review']))
        st.image(wc_neu.to_array(), caption="Neutral Reviews")
    with col3:
        wc_neg = WordCloud(width=500, height=300, background_color='white').generate(
            ' '.join(df_balanced[df_balanced['sentiment'] == 'Negative']['cleaned_review']))
        st.image(wc_neg.to_array(), caption="Negative Reviews")

elif menu == "Model Performance":
    st.header("📈 Model Performance")

    st.subheader("Classification Report")
    st.text(classification_report(y_test, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=model.classes_, yticklabels=model.classes_)
    st.pyplot(fig)
