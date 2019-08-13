# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 14:11:31 2019

@author: chiji
"""

import sys
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    '''
    Loads the datasets into the working directory. This will leave the directry
    with dataframes: messages, categories, and the merged dataframe 'df'
    
    Args:
        messages_filepath- The path where messages dataset is stored
        categories_filepath- The path where categories dataset is stored
    Returns:
        The dataframe containign both messages and categories
    '''
    #Load messages
    mess_path = messages_filepath
    messages = pd.read_csv(mess_path)
    #messages.head()
    
    # load categories dataset
    cat_path = categories_filepath
    categories = pd.read_csv(cat_path)
    #categories.head()
    
    #combine the two
    df = messages.merge(categories, how = 'outer', on = ['id'])
    return df


def clean_data(df):
    '''
    returns a cleaned dataframe of the messages and the categories
    
    Args:
        df- The loaded datafset cotaining the message and the categories
    Returns:
        A dataframe of of the messages and the categories ready for modeling
    '''
    #df = messages.merge(categories, how = 'outer', on = ['id'])
    #categories2 = pd.DataFrame({})
    categories1 = df[['categories']]
    # create a dataframe of the 36 individual category columns
    categories1 = categories1['categories'].str.split(';', expand = True)
    # rename the columns of `categories`
    for col, items in categories1.items():
        categories1.rename(columns = {col:categories1.loc[0,col]}, inplace = True)
    #Iterate through the category columns in df to keep only the last character
    # of each string (the 1 or 0)
    #use_last = lambda x:x[:-1]#Uses all but the last
    use_last = lambda x:x[-1]
    for i,j in categories1.items():
        categories1[i] = categories1[i].apply(use_last)
        categories1[i] = pd.to_numeric(categories1[i])
    #drop the categories column in df
    df.drop('categories', inplace = True, axis = 1)
    
    #use the transformed category as the new category column
    df = pd.concat([df,categories1], axis = 1)
    # remove duplicates
    df4 = df[~pd.DataFrame.duplicated(df, subset = ['id','message'])]
     #categories1.head()
    return df4

def save_data(df, database_filename):
    
    
    '''
    Saves the dataframe 'df' as an sql database file
    Args:
        df- The loaded datafset cotaining the message and the categories
        database_filename - the name of the database to store to
    Returns:
        None
    '''
    conn = sqlite3.connect(database_filename)
    df.to_sql(database_filename, con = conn, if_exists = 'replace', index = False)
    #engine = create_engine('sqlite:///ready_df.db')
    #df.to_sql("ready_df", engine,if_exists = 'replac', index=False)


def main():
        messages_filepath = 'disaster_messages.csv'
        categories_filepath = 'disaster_categories.csv'
        database_filepath = 'DisasterResponse.db'
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()