'''
A machine learning model to categorize a given text message related to natural
disaster into different types of disasters and help authorities respond 
accordingly

Author: Idoko CM 2019

'''
import sys
# import libraries
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sqlalchemy import create_engine
import sqlite3
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import nltk

nltk.download(['punkt', 'wordnet', 'stopwords'])
import re
from nltk.corpus import stopwords

def load_data(database_filepath):
    '''
    Function to load an SQL database into the work environment
    Args:
        database_filepath - The path to the database
    Returns:
        X - the values of the independent variables
        Y - the values of the dependent variables
        Column_names - list of the names of the dependent variables
    '''
    #engine = create_engine()
    conn = sqlite3.connect(database_filepath)
    #df = pd.read_sql('SELECT *FROM "ready_db.db"', engine)
    df = pd.read_sql('SELECT *FROM "DisasterResponse.db"', conn)
    df1 = df[~df['offer-0'].isnull()]
    X = df1[['message']].values
    Y = df1.iloc[:,4:].values
    X = [i for i in X for i in i]
    return X, Y, df1.iloc[:,4:].columns


def tokenize(text):
    '''
    Function to break down a document (sentence) into tokens (words)
    while removing stopwords, regular expressions, white space and punctuations
    Args:
        text - the sentence (document) to tokenize
    Returns:
        A list of words needed to train a machine learning model.
    '''
    text = re.sub(r"[^a-zA-Z0-9]", ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        
        clean_tok = lemmatizer.lemmatize(tok, pos = 'v').lower().strip()
        
        clean_tokens.append(clean_tok)
    clean_tokens = [w for w in clean_tokens if w not in stopwords.words("english")]

    return clean_tokens


def build_model(estimator):
    '''
    Function to build a multi-output predictive model by pipelining through 
    different transformers (including CountVectorizer, and TfidfTransfrmer) using 
    a given estimator.
    Args:
        estimator - an estimator object, such as as RandomForestClassifier(),
        DecisionTreeCalssifier(), etc
    Returns:
        A predictive model which can be trained to predict outputs of same size
        as dependent variables.
    '''
    #df1 = load_data('sqlite:///ready_df.db')
    #estimator =  RandomForestClassifier() # MultinomialNB()
    pipeline = Pipeline([
        ('transformer', Pipeline([
            ('vect', CountVectorizer(tokenizer = tokenize)),
            ('tfidf', TfidfTransformer())
            ])),
         ('clf', MultiOutputClassifier(estimator))
        ])
    #X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)
    
    parameters = { 
    'transformer__vect__max_features': [5000, 3000, 1000],
    'transformer__vect__ngram_range': ((1,1),(1,2)),
    'transformer__tfidf__use_idf': (True, False)
            }
    model = GridSearchCV(pipeline, param_grid = parameters)
    #model.fit(X_train,Y_train)
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    returns a dataframe of the important score metrics of the results of the multi-classification problem
    including the accuracy, precision, recall, and f1-score.
    
    Args:
        model - The model to be evaluated
        X_test - the true values of the independent varaible
        Y_test - The true values of the dependent variable
        y_pred - the predictions of dependent varable using the model
        labels - the list of the names of the columns of the Y_test
    Returns:
        A dataframe of the important score metrics of the results of the multi-classification problem
        including the accuracy, precision, recall, and f1-score
    '''
    score_df = {}
    precision_list = []
    recall_list = []
    f1score_list = []
    accuracy_list = []
    y_pred = model.predict(X_test)
    #labels = df1.iloc[:,4:].columns
    for i in range(Y_test.shape[1]):
        ghy = classification_report(Y_test[:,i], y_pred[:,i], output_dict = True)
        vg =  accuracy_score(Y_test[:,i], y_pred[:,i])
        precision_list.append(ghy['weighted avg']['precision'])
        recall_list.append(ghy['weighted avg']['recall'])
        f1score_list.append(ghy['weighted avg']['f1-score'])
        accuracy_list.append(vg)
    score_df = pd.DataFrame({'Variable':category_names, 'Precision':precision_list,'Recall':recall_list,'F-1_score':f1score_list, 'Accuracy':accuracy_list})
    score_df.set_index('Variable', inplace = True)
    return score_df


def save_model(model, model_filepath):
    '''
    Function to save and export a machine learning (sklearn) model as a pickle
    file into the current filepath
    Args:
        model - the model to save and export
        model_filepath - the path to store model (a string)
    Returns:
        None
    '''
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        #print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        #print('Building model...')
        estimator = RandomForestClassifier()
        model = build_model(estimator)
        
        #print('Training model...')
        model.fit(X_train, Y_train)
        
        #print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        #print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        #print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()