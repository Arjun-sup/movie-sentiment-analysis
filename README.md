### 🌟 Movie Review Sentiment Analysis

This project is a movie review sentiment analysis tool that classifies text as either "Positive" or "Negative." It uses a Logistic Regression model trained on a dataset of 50,000 IMDb movie reviews. The model and the TF-IDF vectorizer are saved to disk, allowing for quick predictions without the need to retrain the model every time the script is run.

### 📁 Repository Structure

```
.
├── main.py
├── IMDB Dataset.csv
├── sentiment_model.joblib
├── tfidf_vectorizer.joblib
└── requirements.txt
```

  * `main.py`: The core Python script for training, loading, and predicting sentiment.
  * `IMDB Dataset.csv`: The dataset used to train the model.
  * `sentiment_model.joblib`: The pre-trained Logistic Regression model.
  * `tfidf_vectorizer.joblib`: The pre-fitted TF-IDF vectorizer.
  * `requirements.txt`: A list of all required Python libraries.

-----

### 🚀 Getting Started

These instructions will get a copy of the project up and running on your local machine.

#### **Prerequisites**

You need to have **Python 3.x** installed on your system.

#### **Installation**

1.  Clone this repository to your local machine:
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd your-repo-name
    ```
3.  Install the necessary Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

-----

### 💻 How to Run

Simply execute the main Python script from your terminal.

```bash
python main.py
```

The script will first check for the pre-trained model files. If they exist, it will load them; otherwise, it will train a new model from the dataset and then save the model files. You will then be prompted to enter a movie review for sentiment prediction.

-----

### 💡 How It Works

The `main.py` script performs the following key steps:

1.  **Data Loading and Cleaning**: It loads the `IMDB Dataset.csv` and prepares the text data by converting it to lowercase, removing HTML tags, and punctuation.
2.  **Training/Loading**: It checks for the `.joblib` model and vectorizer files. If they are found, it loads the pre-trained components. If not, it trains a new **Logistic Regression** model using **TF-IDF features** and saves the model to disk.
3.  **Prediction**: The `predict_sentiment` function takes a new review, transforms it using the TF-IDF vectorizer, and uses the model to predict the sentiment.
4.  **Interactive Interface**: A simple command-line loop allows for continuous sentiment predictions until you type 'exit'.
