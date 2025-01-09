# Sentiment Analysis

This project involves building a sentiment analysis model to classify tweets into positive or negative sentiment. The model is based on logistic regression and achieves an accuracy of *80%* on the validation dataset. Advanced preprocessing techniques, including stemming, emoticon handling, and abbreviation standardization, are employed to improve the efficiency and accuracy of the model.

## Dataset

The dataset used in this project is the [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle. This dataset contains 1.6 million tweets and is commonly used for sentiment analysis tasks.

You can download the dataset from the link below:

- [Sentiment140 Dataset on Kaggle](https://www.kaggle.com/datasets/kazanova/sentiment140)

## Libraries and Tools Used
The following Python libraries are used in this project:

python
import nltk  # Natural Language Toolkit
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import emoji


### Additional Installation:
- *Emoji Library*: Install using the following command:
  bash
  pip install emoji
  

## Dataset
The dataset used is:
- *training.1600000.processed.noemoticon.csv*: A pre-processed dataset of tweets labeled as positive or negative.

## Project Workflow

1. *Data Loading*:
   Load the training.1600000.processed.noemoticon.csv dataset using pandas.

   python
   import pandas as pd
   data = pd.read_csv('training.1600000.processed.noemoticon.csv', encoding='latin1', header=None)
   data.columns = ['Sentiment', 'ID', 'Date', 'Query', 'User', 'Tweet']
   

2. *Data Preprocessing*:
   - Remove unnecessary columns such as ID, Date, Query, and User.
   - Convert text to lowercase.
   - Remove stopwords and punctuation.
   - Perform stemming using PorterStemmer.
   - Handle emoticons and abbreviations using the emoji library.

3. *Feature Extraction*:
   Transform the cleaned tweets into numerical representations using TfidfVectorizer.


4. *Model Training*:
   Split the data into training and validation sets and train the logistic regression model.

   python
   X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
   model = LogisticRegression()
   model.fit(X_train, y_train)
   

5. *Future Goals*:
   - Extend the model to support multilingual sentiment analysis using mBERT.
   - Aim for a global reach with a target accuracy of *75%* for multilingual sentiment classification.
## Usage
1. Install the required libraries:
   bash
   pip install nltk scikit-learn matplotlib emoji pandas
   

2. Place the dataset training.1600000.processed.noemoticon.csv in the project directory.

3. Run the Python script to train the model and evaluate performance.

4. Modify the preprocessing or model parameters to improve performance or adapt the model for new datasets.
