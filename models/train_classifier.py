import sys
from sqlalchemy import create_engine
import pickle
import sqlite3
import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline

nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    """
        Load data
        Load data from database

        Inputs: 
            database_filepath: filepath to database
        Returns:
            df: X, y, category_names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('DataScience', con=engine)
    X = df.message
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    
    return X, y, category_names


def tokenize(text):
    """
        tokenize data
        tokenize data from text

        Inputs: 
            text: text needs to tokenize
        Returns:
            clean_tokens: token after lemmatize
    """
    tokens = word_tokenize(text)
    lemma = WordNetLemmatizer()
    clean_tokens = [lemma.lemmatize(token).lower().strip() for token in tokens]
    return clean_tokens


def build_model():
    """
        build model
        Returns:
            model: GridSearchCV
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    parameters = {
        'clf__estimator__n_estimators' : [10, 20, 30]
    }
    model = GridSearchCV(pipeline, param_grid=parameters)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
        evaluate model
        print classification_report by Y_test and y_pred
        Inputs: 
            model: model trained
            X_test: Data
            Y_test: Data
            category_names: Data
    """
    y_pred = model.predict(X_test)
    for index, column in enumerate(category_names):
        print(column, classification_report(Y_test[column], y_pred[:,index]))
    


def save_model(model, model_filepath):
    """
        save model
        save model to model_filepath

        Inputs: 
            model: model trained
            model_filepath: filepath to save model
    """
    pickle.dump(model, open(model_filepath, 'wb'))



def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()