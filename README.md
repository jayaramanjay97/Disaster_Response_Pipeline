# Disaster Response Pipeline Project
Udacity project: Analyze disaster data from Figure Eight and build a model that classifies disaster messages.

### Libraries Used:
- pandas
- matplotlib
- numpy
- nltk
- sklearn
- sqlalchemy
- pickle

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://localhost:3001/

### References:

The Dataset used for this project is Figure Eight's Multilingual Disaster Response Messages Dataset.

Link  - [Multilingual Disaster Response Messages](https://appen.com/datasets/combined-disaster-response-data)

