# importing 
import os, sys, glob, re
from zipfile import ZipFile
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             make_scorer, plot_confusion_matrix)
import matplotlib.pyplot as plt

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

# importing custom classes
from src import classes as c

def db_create(file_name, database_name):
    """Creates and populates an sqlite database from zipped csv files."""
    
    zip_path = os.path.join(gparent, 'data/raw', file_name)
    out_path  = os.path.join(gparent, 'data/raw')
    db_path =  os.path.join(gparent, 'data/processed', database_name)
    
    # opening the zip file
    with ZipFile(zip_path, 'r') as zip:
        
        # extracting files to raw directory
        zip.extractall(out_path)
        
    # creating and connecting to database
    conn = sqlite3.connect(db_path)  
    
    # creating paths to the files
    path = os.path.join(gparent, 'data/raw', '*.csv')
    ps = []
    for name in glob.glob(path):
        ps.append(name)

    # creating list of tuples with names and data frames; importing data as strings
    dfs = [(re.split('[////./]', file_path)[8], 
            pd.read_csv(file_path, dtype=str)) for file_path in ps]

    # Adding data to the database using the tuples created above.
    # Creating tables from the data frames, and naming the tables with the name strings from above.
    for tup in dfs:
        tup[1].to_sql(tup[0].upper(), conn, if_exists='append', index = False)

def fetch(cur, q):
    """Returns an SQL query."""
    
    return cur.execute(q).fetchall()

def f_score(y_true, y_pred):
    "F1 scoring function for use in make_scorer."
    
    f1 = f1_score(y_true, y_pred)
    return f1

# creating scorer object for pipelines
f1 = make_scorer(f_score)

def recall(y_true, y_pred):
    "Recall scoring function for use in make_scorer."
    
    rec = recall_score(y_true, y_pred)
    return rec

# creating scorer object for cv
recall = make_scorer(recall)

def precision(y_true, y_pred):
    "Precision scoring function for use in make_scorer."
    
    pre = precision_score(y_true, y_pred)
    return pre

# creating scorer object for cv
precision = make_scorer(precision)

def splitter(X, y):
    """Returns a train/test split."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=2021,
                                                        stratify=y
                                                   )
    return  X_train, X_test, y_train, y_test

def confusion_report(model, X, y):
    "Returns a confusion matrix plot and scores."
    
    f1 = f1_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    precision = precision_score(y, model.predict(X),)
    report = pd.DataFrame([[f1, recall, precision]],\
                          columns=['F1', 'Recall', 'Precision']) 
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap=plt.cm.Blues, 
                          display_labels=['Positive', 'Negative'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
#     plt.savefig('dummy',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()  
    
    return report 
      
