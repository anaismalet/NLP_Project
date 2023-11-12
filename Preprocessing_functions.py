
""" 

NLP project 

Author : Anaïs Malet
Date of creation : 16/10/2023

This python file contains usefull function.

"""
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import re
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

'''
Preprocessing function that performs several text preprocessing steps using the Natural Language Toolkit (NLTK) library to clean and prepare text data.

The function :

- Lemmatize the text : a text normalization technique that reduces words to their base or root form.
- Initializes a set of stopwords for the English language using NLTK's stopwords.words('english'
- Tokenizes the input text into a list of words or tokens using NLTK's word_tokenize function.
- It converts each word to lowercase and filters out non-alphanumeric tokens (keeps only alphanumeric words).
- Filters out words that are present in the stop words set, removing common words that typically don’t add much information to the text.
- Lemmatizes each word in the filtered word list, reducing them to their base form.

and then joins the processed words back into a single string.

'''
def preprocess_text(text):
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    words = [word.lower() for word in tokens if word.isalnum()]
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words)
