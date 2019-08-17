# Load modules
import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Inputs:
      database_filepath - Filepath to SQL database of disaster response messages with categorizations to use for machine learning model
    Outputs:
      x - Numpy series of disaster response messages (machine learning model input)
      y - pandas dataframe of disaster response message categorizations (machine learning model output)
      category_names - List of names of disaster response message categorizations
    """
            
    # Create SQL engine name
    engine_name = 'sqlite:///' + database_filepath
    
    # Create engine object
    engine = create_engine(engine_name)
    
    # Extract database name from database_filepath
    database_name = database_filepath.split('/')[-1].split('.')[0]
    
    # Create dataframe from SQL database
    df = pd.read_sql(database_name, engine)
    
    # Extract the disaster response messages from dataframe
    x = df.message
    
    # Extract disaster response message categorization category names from dataframe
    category_names = df.columns[4:]
    
    # Extract disaster response message categorizations from dataframe
    y = df[category_names]
       
    return x, y, category_names


def tokenize(text):
    """
    Inputs:
      text - A disaster response messages (passed as a string)
    Outputs:
      clean_tokens - Cleaned natural language processing tokens of input string
    """
       
    # Remove Punctuation Characters
    text_no_punc = re.sub(r"[^a-zA-Z0-9]", " ", text) 
       
    # Tokenize the text
    tokens = word_tokenize(text_no_punc)
    
    # Remove stop words from the text
    filtered_tokens = [token for token in tokens if token not in stopwords.words("english")]
    
    # Initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # Lemmatize and normalize the tokens
    clean_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in filtered_tokens]
        
    return clean_tokens


def build_model():
    """
    Inputs:
      None
    Outputs:
      drm_ml_model - Machine learning model to be used for classifying disaster response message categories
    """
    
    # Build pipeline to vectorize text, transform it, and classify all 36 categories using Random Forest
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs = -1))
    ])
    
    # Define parameters for pipeline hyperparameter grid search
    parameters = {
        'clf__estimator__n_estimators': [100, 200]
    }
    
    # Create grid search object
    drm_ml_model = GridSearchCV(pipeline, param_grid=parameters)
    
    return drm_ml_model


def evaluate_model(model, x_test, y_test, category_names):
    """
    Inputs:
      model - Trained natural language processing machine learning model
      x_test - disaster response messages for testing the machine learning model
      y_test - disaster response message categorizations for testing the machine learning model
      category_names - disaster response message categorization category names to use for evaluating the machine learning model
    Outputs:
      Prints the machine learning classification report for the input columns using the test inputs and test responses
    """
    
    # Create category predictions for test disaster response messages
    y_pred = model.predict(x_test)
    
    # Print classification report for each provided category name
    for category_index, category_name in enumerate(category_names):
        print("Classification Report for Category '{}'\n".format(category_name))
        print(classification_report(y_test[category_name], y_pred[:, category_index]))
        print('-----------------------------------------------------')


def save_model(model, model_filepath):
    """
    Inputs:
      model - Trained machine learning model to be used for classifying disaster response message categories
      model_filtepath - Name of pickle file to save trained machine learning model to
    Outputs:
      Saves trained machine learning model to a pickle file
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