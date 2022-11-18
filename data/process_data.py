import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Args:
        messages_filepath (str): path to the messages csv file
        categories_filepath (str): path to the categories csv file
        
    Returns:
        df (DataFrame): merged dataframe
    """

    messages = pd.read_csv(messages_filepath)
    print('Messages Shape:', messages.shape)
    categories = pd.read_csv(categories_filepath)
    print('Categories Shape:', categories.shape)
    data = messages.merge(categories, on='id')
    print('Message and Categories merged', data.info())
    
    return data
    


def clean_data(data):
    """
    Args:
        data (DataFrame): the dataframe to be cleaned
    Returns:
        data (DataFrame): the cleaned data
    """

    categories = data.categories.str.split(';', expand=True) #split categories cloumn

    first_row = categories.iloc[0]
    print('Categories : ', first_row)
    
    # extract a list of new column names for categories.
    column_names = first_row.apply(lambda x : x[:-2])
    categories.columns = column_names
    print('categories columns: ', categories.columns)
    
    for column in categories:
        categories[column] = categories[column].str[-1].astype(int) # convert category values to binary

    data.drop('categories', axis=1, inplace=True) 
    data = pd.concat([data, categories], axis=1) 
    data.drop_duplicates(inplace=True)
    data = data[data['related'] != 2] # remove non-binary values
    
    return data



def save_data(df, database_filename):
    """save the dataframe to a sqlite database 
    Args:
        df (DataFrame): the dataframe to be saved
        database_filename (str): the name of the database
        
     Returns: saved data in database
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False, if_exists='replace')


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