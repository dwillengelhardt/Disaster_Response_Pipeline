# Disaster Response Pipeline Project

## by Dave Engelhardt

### Project Summary
In this project, disaster data from Figure Eight is used to build a machine learning model for an API that classifies disaster messages.  The project includes a web app where an emergency service worker can input a new message and get classification results in several categories.  The web app also displays visualizations of the data.

### Files
Below is the file structure for the project, along with file descriptions for each of the files

- app
| - template
| |- master.html  - Main page of web app
| |- go.html  - Classification result page of web app
|- run.py  - Flask file that runs web app

- data
|- disaster_categories.csv  - Disaster response message categorization data to process and use in creating machine learning model
|- disaster_messages.csv  - Disaster response message data to process and use in creating machine learning model
|- process_data.py - Script to process disaster message and categorization data and store it in an SQL database
|- InsertDatabaseName.db - SQL database where processed disaster message and categorization data is stored

- models
|- train_classifier.py - Script to use processed data and build a trained machine learning model for the web app
|- classifier.pkl - Saved trained machine learning model for the web app

- README.md

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in an SQL database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
        
    - To run ML pipeline that trains classifier and saves it
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run the web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/ to interact with the web app
