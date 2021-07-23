import pandas as pd
import numpy as np

from sklearn.model_selection import (train_test_split, cross_validate)
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             make_scorer)

import os, sys, glob, re
from zipfile import ZipFile
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import Image

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

class Harness:
    
    def __init__(self, scorer, random_state=2021):
        self.scorer = scorer
        self.history = pd.DataFrame(columns=['Name', 'F1', 'Recall',
                                             'Precision', 'Notes'])

    def report(self, model, X, y, name, notes='', cv=5,):
        self.scores = cross_validate(model, X, y, 
                                 scoring=self.scorer, cv=cv)
        frame = pd.DataFrame([[name, self.scores['test_f1'].mean(),
                               self.scores['test_recall'].mean(),
                               self.scores['test_precision'].mean(), notes]],
                             columns=['Name', 'F1', 'Recall',
                                      'Precision', 'Notes'])
        self.history = self.history.append(frame)
        self.history = self.history.reset_index(drop=True)
        self.history = self.history.sort_values('F1')
        self.print_error(name, self.scores['test_f1'].mean())
        return [self.scores['test_f1'].mean(), self.scores['test_recall'].mean(),
                self.scores['test_precision'].mean()]

    def print_error(self, name, Accuracy):
        print(f"{name} has an average F1 of {self.scores['test_f1'].mean()}")
        print(f"{name} has an average Recall of {self.scores['test_recall'].mean()}")
        print(f"{name} has an average Precision of {self.scores['test_precision'].mean()}")

class Database:
    
    def __init__(self, database_name):
        """Creates database, cursor, and connects to database."""
         
        data_path = os.path.join(gparent,'data/processed','outcomes.db')
        self.conn = sqlite3.connect(data_path)  
        self.cur = self.conn.cursor()
        
    def populate(self, file_name, database_name):
        """Populates an sqlite database from zipped csv files."""

        zip_path = os.path.join(gparent, 'data/raw', file_name)
        out_path  = os.path.join(gparent, 'data/raw')
        db_path =  os.path.join(gparent, 'data/processed', database_name)
        # opening the zip file
        with ZipFile(zip_path, 'r') as zip:
            # extracting files to raw directory
            zip.extractall(out_path)
        # creating paths to the files
        path = os.path.join(gparent, 'data/raw', '*.csv')
        self.ps = []
        for name in glob.glob(path):
            self.ps.append(name)
        # creating list of tuples with names and data frames; importing data as strings
        self.dfs = [(re.split('[////./]', file_path)[8], 
                pd.read_csv(file_path, dtype=str)) for file_path in self.ps]
        # Adding data to the database using the tuples created above.
        # Creating tables from the data frames, and naming the tables with the name strings from above.
        for tup in self.dfs:
            tup[1].to_sql(tup[0].upper(), self.conn, if_exists='append', index = False)
            
    def fetch(self, cur, q):
        """Returns an SQL query."""
        
        return self.cur.execute(q).fetchall()
    
    def table_names(self):
        """Returns the table names of the database."""
        
        q ="""
        SELECT name 
        FROM sqlite_master 
        WHERE type IN ('table','view') 
        AND name NOT LIKE 'sqlite_%'
        ORDER BY 1
        """
        tables = self.fetch(self.cur, q)
        tables = [t[0] for t in tables]
        return tables
    
    def simple_df(self, table_name):
        """Returns data frame made from entries in table"""
        
        q = f"SELECT*FROM {table_name}"
        df = pd.DataFrame(self.fetch(self.cur, q))
        df.columns = [i[0] for i in self.cur.description]
        return df

    def table_info(self, table_name):
        """Return the column names and info from the database."""
        
        # getting column names and table info
        q= f"PRAGMA table_info({table_name})"
        return self.fetch(self.cur, q)
            
    # month codes used in code_presentation
    month_codes = {
        'A': 'January', 
        'B': 'Februrary',
        'C': 'March', 
        'D': 'April',
        'E': 'May',
        'F': 'June',
        'G': 'July',
        'H': 'August',
        'I': 'Septemeber',
        'J': 'October',
        'K': 'November',
        'L': 'December'
            }

    # module codes and summaries
    module_codes = {
        'AAA': {'Domain': 'Social Sciences', 'Presentations': 2, 'Students': 748},
        'BBB': {'Domain': 'Social Sciences', 'Presentations': '4', 'Students': 7909},
        'CCC': {'Domain': 'STEM', 'Presentations': 2, 'Students': 4434},
        'DDD': {'Domain': 'STEM', 'Presentations': 4, 'Students': 6272},
        'EEE': {'Domain': 'STEM', 'Presentations': 3, 'Students': 2934},
        'FFF': {'Domain': 'STEM', 'Presentations': 4, 'Students': 7762},
        'GGG': {'Domain': 'Social Sciences', 'Presentations': 3, 'Students': 2534}
    }
    
    # edr 
    path = os.path.join(gparent,'references','schema.png')
    
    erd = Image(filename=path) 

