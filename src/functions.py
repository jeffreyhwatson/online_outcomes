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

data_path = os.path.join(gparent,'data/processed','outcomes.db')
conn = sqlite3.connect(data_path)  
cur = conn.cursor()

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
    """Returns a confusion matrix plot and scores."""
    
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

def binarize_target(df):
    """Binarizes target and moves column to front of data frame."""
    
    binarize = {'Pass': 1, 'Withdrawn': 0, 'Fail': 0, 'Distinction': 1}
    df['target'] = df['final_result']
    df['target'] = df['target'].map(binarize)
    col_name = 'target'
    first = df.pop(col_name)
    df.insert(0, col_name, first)
    return df

def df_fixes(df):
    """Performs various fixes to the dataframe."""
    
    # dropping duplicate columns
    df = df.T.drop_duplicates().T
    # moving final_result to front of df
    col = 'final_result'
    first_col = df.pop(col)
    df.insert(0, col, first_col)
    df.head()
    # dropping nulls
    df.dropna(inplace=True)
    # fixing typo
    df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
    # renaming values
    df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
    df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
    # converting datatypes
    conversions = ['click_sum', 'date', 'num_of_prev_attempts','studied_credits']
    df[conversions] = df[conversions].apply(pd.to_numeric)
    # adding course_load column
    df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                  labels=['Light', 'Medium', 'Heavy'],\
                                  duplicates='drop')
    return df

def sv_si():
    """Making new df by joining studentinfo and studentvle tables
       and creating a click_sum column."""
    
    q = """
    SELECT SV.*, 
    SUM(SV.sum_click) AS click_sum,
    SI.*
    FROM 
    STUDENTVLE as SV
    JOIN 
    STUDENTINFO as SI
    ON SV.code_module = SI.code_module
    AND SV.code_presentation = SI.code_presentation
    AND SV.id_student = SI.id_student
    GROUP BY 
    SV.code_module,
    SV.code_presentation,
    SV.id_student;
    """
    df = pd.DataFrame(fetch(cur, q))
    df.columns = [i[0] for i in cur.description]
    return df_fixes(df) 