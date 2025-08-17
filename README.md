Project Title

Movie Review Sentiment Analysis



Description

This project is a movie review sentiment analysis tool that classifies text as either "Positive" or "Negative." It uses a Logistic Regression model trained on a dataset of 50,000 IMDb movie reviews. The model and the TF-IDF vectorizer are saved to disk, allowing for quick predictions without the need to retrain the model every time the script is run.



Files in this Repository

main.py: The main Python script for training, loading, and predicting sentiment.



IMDB Dataset.csv: The dataset used to train the model.



sentiment\_model.joblib: The pre-trained Logistic Regression model.



tfidf\_vectorizer.joblib: The pre-fitted TF-IDF vectorizer.



Getting Started

These instructions will get a copy of the project up and running on your local machine.



Prerequisites

You need to have Python installed.



Installation

Clone this repository to your local machine:

git clone https://github.com/your-username/your-repo-name.git



Navigate to the project directory:

cd your-repo-name



Install the required Python libraries using the requirements.txt file:

pip install -r requirements.txt



How to Run

Simply execute the Python script from your terminal:

python main.py



The script will first check for the pre-trained model files.



If they exist, it will load them and prompt you to enter a movie review for sentiment prediction.



If they don't exist, it will train a new model, save the .joblib files, and then prompt you for input.



How It Works

The main.py script performs the following steps:



Data Loading and Cleaning: It loads the IMDB Dataset.csv and cleans the text by converting it to lowercase, removing HTML tags, and removing punctuation.



Training/Loading: It checks for the sentiment\_model.joblib and tfidf\_vectorizer.joblib files.



If they are found, it loads the pre-trained model and vectorizer.



If they are not found, it trains a new Logistic Regression model using TF-IDF features and saves the model and vectorizer to disk for future use.



Prediction: The predict\_sentiment function cleans and vectorizes a new review, then uses the loaded model to predict the sentiment (positive or negative).



User Interface: A simple loop allows the user to enter multiple reviews for prediction until they type 'exit'.

