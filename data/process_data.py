# Load modules
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Inputs:
      messages_filepath - Filepath to disaster response message CSV file (machine learning algorithm input)
      categories_filepath - Filepath to disaster response message categorizations CSV file (machine learning algorithm output)
    Outputs:
      df_combined - pandas dataframe of data in disaster response message CSV file merged with categorizations CSV file
    """
    
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge messages and categories dataset
    df_combined = messages.merge(categories, on = 'id', suffixes = ('_msg', '_cat'))
    
    return df_combined


def clean_data(df):
    """
    Inputs:
      df - pandas dataframe of data in disaster response message CSV file merged with categorizations CSV file
    Outputs:
      df_clean - Cleaned version of input (duplicates are removed and categories are split into labeled, individual columns)
    """
    
    # Create dataframe of individual column categories
    categories = df.categories.str.split(pat = ';', expand = True)
    
    # Extract column names for categories
    category_col_names = [item[0:-2] for item in categories.iloc[0]]
    
    # Rename the columns of categories dataframe
    categories.columns = category_col_names
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
        
    # Drop the original categories column from df
    df.drop(['categories'], axis = 1, inplace=True)
    
    # Concatenate the original dataframe with the new categories dataframe
    df = df.join(categories)
    
    # Remove duplicates from concatenated dataframe
    df_clean = df.drop_duplicates()
    
    return df_clean


def save_data(df, database_filename):
    """
    Inputs:
      df - Cleaned version of pandas dataframe of data in disaster response message CSV file merged with categorizations CSV file
      database_filename - Name of SQL database to use to externally save df
    Outputs:
      Saves SQL database of input dataframe
    """
    
    # Create SQL engine name
    engine_name = 'sqlite:///' + database_filename
    
    # Create engine object
    engine = create_engine(engine_name)
    
    # Extract database name from database_filepath
    database_name = database_filename.split('/')[-1].split('.')[0]
    
    # Save dataframe to SQL database
    df.to_sql(database_name, engine, index=False)


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