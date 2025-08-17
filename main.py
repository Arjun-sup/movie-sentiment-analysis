import pandas as pd
import os 
from joblib import dump,load
df =pd.read_csv("IMDB Dataset.csv")

print("\n First 5 rows")
print(df.head())

print(df.shape)

print("\n Missing Values:\n", df.isnull().sum())
print("\n Sentiment Distribution:\n", df['sentiment'].value_counts())

import re
#Function to clean text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"<.*?>","",text)#remove HTML tags
    text=re.sub(r"[^a-zA-Z\s]","",text)
    text=re.sub(r"\s+"," ",text).strip()#remove extra spaces
    return text



#show example cleaned review

#print("\n Sample Cleaned Review:\n", df['cleaned_review'][0])

from sklearn.feature_extraction.text import TfidfVectorizer




from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# File paths to save/load
MODEL_FILE = "sentiment_model.joblib"
VECTORIZER_FILE = "tfidf_vectorizer.joblib"

# Always clean the text, regardless of training or loading
print("\n Cleaning text column...")
df['cleaned_review'] = df['review'].apply(clean_text)


if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    print("\n Loading model and vectorizer from disk...")
    model = load(MODEL_FILE)
    vectorizer = load(VECTORIZER_FILE)
else:
    print("\n Training model and saving to disk...")

    #Apply Cleaning function to all reviews
    print("\n Cleaning Text")
    df['cleaned_review']=df['review'].apply(clean_text)

    # Create TF-IDF vectorizer (again, same setup)
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(df['cleaned_review'])
    #Labels
    y=df['sentiment']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print(" Accuracy:", accuracy_score(y_test, y_pred))

    # Save model and vectorizer
    dump(model, MODEL_FILE)
    dump(vectorizer, VECTORIZER_FILE)
    #Print feature shape
    print("\nTF-IDF feature matrix shape:", X.shape)

if not os.path.exists(MODEL_FILE):
    X = vectorizer.fit_transform(df['cleaned_review'])  # Already exists inside else, so skip this here
else:
    X = vectorizer.transform(df['cleaned_review'])  # Needed after loading

def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])  # transform into TF-IDF vector
    prediction = model.predict(vectorized)[0]     # 0 = negative, 1 = positive
    return "Positive " if prediction == 1 else "Negative "

example = "I absolutely loved this movie!"
print(f"\nExample Review: {example}\nPredicted Sentiment: {predict_sentiment(example)}")

while True:
    user_input = input("\nEnter a movie review (or type 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    sentiment = predict_sentiment(user_input)
    print("Predicted Sentiment:", sentiment)