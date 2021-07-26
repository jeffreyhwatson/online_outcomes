import pandas as pd
import numpy as np
import os, sys, glob, re
from zipfile import ZipFile
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import Image

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

class Database:
    
#    init method
    
    def __init__(self, database_name):
        """Creates database, cursor, and connects to database."""
         
        data_path = os.path.join(gparent,'data/processed','outcomes.db')
        self.conn = sqlite3.connect(data_path)  
        self.cur = self.conn.cursor()
        
#   attributes  
    
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
    
    # erd image
    path = os.path.join(gparent,'references','schema.png')
    
    erd = Image(filename=path) 
    
#   methods    
        
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
        """Returns data frame made from all entries in table."""
        
        q = f"SELECT*FROM {table_name}"
        df = pd.read_sql(q, self.conn)
        return df

    def table_info(self, table_name):
        """Return the column names and info from the database."""
        
        # getting column names and table info
        q= f"PRAGMA table_info({table_name})"
        return self.fetch(self.cur, q)

    
    def binarize_target(self, df):
        """Binarizes target and moves column to front of data frame."""
    
        binarize = {'Pass': 1, 'Withdrawn': 0, 'Fail': 0, 'Distinction': 1}
        df['target'] = df['final_result']
        df['target'] = df['target'].map(binarize)
        col_name = 'target'
        first = df.pop(col_name)
        df.insert(0, col_name, first)
        return df
    
    def si_fixes(self, df):
        """Performs various fixes to the dataframe."""
    
        # dropping duplicate columns
        df = df.T.drop_duplicates().T
        # moving final_result to front of df
        col = 'final_result'
        first_col = df.pop(col)
        df.insert(0, col, first_col)
        # dropping nulls
        df = df.dropna()
        # fixing typo
        df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
        # renaming values
        df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
        df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
        # converting datatypes
        conversions = ['studied_credits']
        df[conversions] = df[conversions].apply(pd.to_numeric)
        # adding course_load column
        df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                      labels=['Light', 'Medium', 'Heavy'],\
                                      duplicates='drop')
        # binarizing target
        df = self.binarize_target(df)
        # dropping extraneous columns
        df = df.drop(columns=['final_result', 'studied_credits'])
        return df

    def student_info(self):
        """Returns dataframe from the STUDENTINFO table with various fixes."""
        
        q = "SELECT*FROM STUDENTINFO"
        df = pd.read_sql(q, self.conn)
        return self.si_fixes(df)
    
    def sv_fixes(self, df):
        """Performs various fixes to the dataframe."""
    
        # converting datatypes
        conversions = ['click_sum', 'date', 'num_of_prev_attempts']
        df[conversions] = df[conversions].apply(pd.to_numeric)
        # dropping extraneous column
        df = df.drop(columns=['sum_click'])
        return df

    def sv_si(self):
        """
        Making df joining studentinfo & studentvle and creating a click_sum column.
        """
    
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
        df = pd.read_sql(q, self.conn)
        df = self.si_fixes(df)
        df = self.sv_fixes(df)        
        return df