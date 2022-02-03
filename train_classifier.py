import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import re
import pickle

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, f1_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def load_data(database_filepath):
    """ Load dataset from database & define feature and target variables X and Y
    
    Args: 
    database_filepath (str): database's filepath
    
    Returns:
    X: dataframe with input variable 'message'
    y: dataframe with target variables (categories)
    categories_names (list): names of categories
        
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("df", engine)
    X=df.message #feature
    y=df.drop(columns=['id','message','original','genre'], axis=1) #target
    cat_names=y.columns
    return X,y, cat_names




def tokenize(text):
    """Tokenize function using nltk to case normalize, lemmatize, and tokenize text.
    
    Args: 
    text (str)
    
    Returns:
    clelan_tokens (list): list of clean key words
    
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    stop_words = stopwords.words("english")
    
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        if tok not in stop_words:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline=Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (NumPy array or pandas series)
    preds - the predictions for those values from some model (NumPy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements 
    
    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    # predict on test data
    preds=model.predict(X_test)
    accuracy=[]
    precision=[]
    recall=[]
    f1=[]
    for i, col in enumerate(category_names):
        accuracy.append(accuracy_score(Y_test.iloc[:,i], preds[:,i]))
        precision.append(precision_score(Y_test.iloc[:,i], preds[:,i], average='weighted'))
        recall.append(recall_score(Y_test.iloc[:,i], preds[:,i], average='weighted'))
        f1.append(f1_score(Y_test.iloc[:,i], preds[:,i], average='weighted'))
        
    print('\nAccuracy score\n', accuracy)
    print('\nPrecision score\n', precision)
    print('\nRecall score\n ',recall)
    print('\nF1 score\n',f1)
    print('\n\n')    
    
    return accuracy, precision, recall, f1
   


def save_model(model, model_filepath):
    """ Save model into a pickle file
    Args:
    model to save
    model_filename
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