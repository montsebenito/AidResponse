import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ Load & Merge Datasets 
        
        Args:
        categories_filepath (str): categories file's path
        messages_filepath (str): messages file's path
        
        Returns:
        df (pandas.DataFrame): dataframe containing the merged uncleaned dataset"""
    
    messages=pd.read_csv(messages_filepath)
    categories=pd.read_csv(categories_filepath)
    df=messages.merge(categories, on='id')
    return df;


def clean_data(df):
    """ Split categories into different columns & convert categories values to 0 or 1
    
    Args:
    df (pandas.DataFrame): dataframe containing the merged uncleaned dataset
    
    Returns:
    df (pandas.DataFrame): cleaned dataframe"""
    
    categories=df.categories.str.split(';', expand=True)
    row=categories.iloc[0]
    categories.columns=row.apply(lambda x: x[:-2])
    for column in categories:
        categories[column]=categories[column].astype(str).apply(lambda x: x[-1])
        categories[column] = categories[column].astype(int)
    df.drop(columns=['categories'], inplace=True)
    df=pd.concat([df,categories],sort=False, axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """ Save data into a database
    Args:
    dataframe to save
    database filename
    """
    
    engine=create_engine('sqlite:///'+database_filename)
    df.to_sql('df', engine, index=False,if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()