""" 

NLP project 

Author : Anaïs Malet
Date of creation : 16/10/2023

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
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


'''

We'll create a class Model in order to test different Vectorizer and Model architecture. 
In this class we'll define a pipeline. This pipeline will take raw reviews as input, preprocess and vectorize them, before fitting a classification model to it.
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

'''
class Model:

    ''' Initialize parameters '''
    
    def __init__(self, X, y, model_architecture, vectorizer, random_seed, test_size) -> None:
        
        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.random_seed = random_seed
        self.test_size = test_size

        
        self.pipeline = Pipeline([
        ("Vectorizer", self.vectorizer),
        ("Model_Architecture", self.model_instance)
        ])

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)# train test split using the above X, y, test_size and random_state


    '''
    Preprocessing the data

    Here's the preprocessing steps we'll use :

    * Removing twitter handles and url
    * Tokenizing the text
    * Lowering and removing non alphanumeric characters
    * Removing stopwords
    * Lemmatizing

    '''
    def preprocess(self, text):
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        words = [word.lower() for word in tokens if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    ''' Training '''

    def fit(self):
        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    ''' Predictions '''

    def predict(self):
        return self.pipeline.predict(self.X_test)
    
    
    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)

    ''' Results visualization '''

    def report(self, class_labels):
        # the report function as defined previously
        print(classification_report(self.y_test, self.predict(), target_names=class_labels))
        confusion_matrix_f = confusion_matrix(self.y_test, self.predict())
        # styling the confusion matrix
        fig = px.imshow(
            confusion_matrix_f, 
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues'  
            )
        fig.show()

'''

We'll create a improved class of the class Model in order to add a undersampler or oversampler to our pipeline. 
In this class we'll define a pipeline. This pipeline will take raw reviews as input, preprocess and vectorize them, before fitting a classification model to it.
The purpose of the pipeline is to assemble several steps that can be cross-validated together while setting different parameters.

'''

class Improved_model:

    ''' Initialize parameters '''
    
    def __init__(self, X, y, model_architecture, vectorizer, sampling, random_seed, test_size) -> None:
        
        self.X = X
        self.y = y
        self.model_instance = model_architecture
        self.vectorizer = vectorizer
        self.sampling = sampling
        self.random_seed = random_seed
        self.test_size = test_size

        
        self.pipeline = make_pipeline(
            self.vectorizer,                        
            self.sampling,  
            self.model_instance                    
        )

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_size, random_state=self.random_seed)# train test split using the above X, y, test_size and random_state


    '''
    Preprocessing the data

    Here's the preprocessing steps we'll use :

    * Removing twitter handles and url
    * Tokenizing the text
    * Lowering and removing non alphanumeric characters
    * Removing stopwords
    * Lemmatizing

    '''
    def preprocess(self, text):
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        tokens = word_tokenize(text)
        words = [word.lower() for word in tokens if word.isalnum()]
        words = [word for word in words if word not in stop_words]
        words = [lemmatizer.lemmatize(word) for word in words]
        
        return ' '.join(words)
    
    ''' Training '''

    def fit(self):
        # fit self.pipeline to the training data
        self.pipeline.fit(self.X_train, self.y_train)

    ''' Predictions '''

    def predict(self):
        return self.pipeline.predict(self.X_test)
    

    def predict_proba(self):
        return self.pipeline.predict_proba(self.X_test)

    ''' Results visualization '''

    def report(self, class_labels):
        # the report function as defined previously
        print(classification_report(self.y_test, self.predict(), target_names=class_labels))
        confusion_matrix_f = confusion_matrix(self.y_test, self.predict())
        # styling the confusion matrix
        fig = px.imshow(
            confusion_matrix_f, 
            text_auto=True, 
            title="Confusion Matrix", width=1000, height=800,
            labels=dict(x="Predicted", y="True Label"),
            x=class_labels,
            y=class_labels,
            color_continuous_scale='Blues'  
            )
        fig.show()