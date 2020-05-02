import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk,re
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV



class Length(BaseEstimator, TransformerMixin):
    """
    To Add the length of message as a Feature
    """
    
    def length(self,text):
        
        return len(text)
    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_len = pd.Series(X).apply(self.length)
        return pd.DataFrame(X_len)
    
def load_data(database_filepath):
    """
    Load Data from the Database
    
    Parameters:
    database_filepath - Path to SQL Database
    
    Returns:
    X - Features Dataframe
    Y - Categories Dataframe
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Dataset',engine)
    X =df['message']
    Y = df[df.columns[4:]]
    return X,Y,Y.columns.tolist()

def tokenize(text):
    """
    Tokenize the text
    Parameters:
    text - Text to be tokenized
    Returns:
    lemmed - Lemmatized tokens
    """
    text = re.sub("[^A-z0-9]+"," ",text)
    text = text.lower()
    words = word_tokenize(text)
    words= [w for w in words if w not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w).strip() for w in words]
    
    return lemmed


def build_model():
    """
    Build a Pipeline Function
    
    Returns:
    cv - Best Model
    """
    parameters = {
        
        'clf__estimator__n_estimators': [50,100],
        'clf__estimator__min_samples_split': [2,3],
      
    } 


    pipeline = Pipeline([
    ('features',FeatureUnion([
        ('text_pipeline',Pipeline([
            ('vect',CountVectorizer(tokenizer=tokenize)),
            ('tfidf',TfidfTransformer(smooth_idf=False)),
            
        ])),
        ('Length',Length())
        
    ])),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    
    ])
    
    cv =  GridSearchCV(pipeline, param_grid=parameters,cv=2)
    return cv

def report(y,yhat,classes):
    """
    Displays classification report for all categories
    
    Parameters:
    y - list of labels
    yhat - list of predictions
    classes - list of categories
    
    
    """
    results = []
    for i in range(y.shape[1]):
        t =(classification_report(y.iloc[:,i],yhat[:,i]))
        f1=(float(t[-15:-11]))#f1
        recall=(float(t[-25:-21]))#recall
        precision=(float(t[-35:-31]))#precision
        results.append([classes[i],precision,recall,f1])
    results.append(['Average',np.average(precision),np.average(recall),np.average(f1)])
    return pd.DataFrame(results,columns=['Classes','Precision','Recall','F1']).set_index('Classes')

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate a Model Function
    
    Parameters:
    model - Best Model
    X_test - Test features
    Y_test - Test Label
    category_names - list of categories
    """
    y_pred = model.predict(X_test)

    cfr = report(Y_test,y_pred,category_names)
    print(cfr)


def save_model(model, model_filepath):
    """
    Save Pipeline Function
    Parameters:
    model - Model to be saved
    model_filepath - destination path to save model as .pkl file
    """
    with open(model_filepath,'wb') as f:
        pickle.dump(model,f)


def main():
    """
    Main function to train classifier
    
    """
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