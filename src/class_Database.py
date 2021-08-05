import pandas as pd
import numpy as np
import os, sys, glob, re
from zipfile import ZipFile
import sqlite3
import matplotlib.pyplot as plt
from IPython.display import Image
from src import helper_functions as f

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

    def student_info(self):
        """Returns dataframe from the STUDENTINFO table with various fixes."""
        
        q = """
        SELECT SI.*,
        /* creating the row_id column by concatenation*/
        SI.code_module || SI.code_presentation || SI.id_student AS row_id,
        /* creating binarized target column*/
        iif(SI.final_result='Pass' OR SI.final_result='Distinction', 0, 1) AS target
        FROM STUDENTINFO AS SI"""
        df = pd.read_sql(q, self.conn)
        return df
    
    def pipe_cleaner(self, df, col_list):
        def drop_cols(df, col_list):
            df = df.drop(col_list, axis=1)
            return df
        def null_zap(df):
            df = df.dropna()
            return df
        def replace_vals(df):
            # fixing typo
            df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
            # renaming values
            df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
            df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
            return df
        def drop_outliers_sc(df):
            Q1 = df.studied_credits.quantile(0.25)
            Q3 = df.studied_credits.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.studied_credits < (Q1 - 1.5 * IQR))\
                                  |(df.studied_credits > (Q3 + 1.5 * IQR)))].copy()
            return df
        df = (df.pipe(replace_vals).pipe(null_zap).pipe(drop_outliers_sc)
              .pipe(drop_cols, col_list))
        return df
     
    def sv_fixes(self, df):
        """Performs various fixes to the dataframe."""
    
        # converting datatypes
        conversions = ['click_sum', 'num_activities','num_of_prev_attempts']
        df[conversions] = df[conversions].apply(pd.to_numeric)
        return df

    def sql_fixes(self, df):
        """Performs various fixes to the dataframe."""
    
        # moving target & row_id to front of df
        f.col_pop(df, 'target')
        f.col_pop(df, 'row_id', 1)
        conversions = ['target', 'studied_credits']
        df[conversions] = df[conversions].apply(pd.to_numeric)
        # adding course_load column
        df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                      labels=['Light', 'Medium', 'Heavy'],\
                                      duplicates='drop')
        return df

    def sv_si(self):
        """
        Making df joining studentinfo & studentvle and creating a click_sum column.
        """
    
        q = """
        SELECT 
        SV.*,
        SI.*,
        /* creating the row_id column by concatenation*/
        SV.code_module || SV.code_presentation || SV.id_student AS row_id,
        /* creating the click_sum column*/
        SUM(SV.sum_click) AS click_sum,
        /* creating the num_activities column*/
        COUNT(SV.sum_click) AS num_activities,        
        /* creating binarized target column*/
        iif(SI.final_result='Pass' OR SI.final_result='Distinction', 0, 1) AS target
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
        # dropping duplicate columns
        df = df.T.drop_duplicates().T
        df = self.sql_fixes(df)
        df = self.sv_fixes(df)

        return df
    
    def df_a(self):
        # creating SVLE & SINFO df
        sv_si = self.sv_si()
        #creating ASSESSMENT df
        assess_df = self.simple_df('ASSESSMENTS')
        #creating STUDENTASSESSMENT df
        stuassess = self.simple_df('STUDENTASSESSMENT')
        # converting score DType
        stuassess['score'] = stuassess['score'].apply(pd.to_numeric)
        # getting each student's mean score
        mean_scores = stuassess.groupby(['id_student'])['score'].mean()\
        .reset_index(name='mean_score')
        #merging the dfs
        sv_si = sv_si.merge(mean_scores, on='id_student')
        # getting each student's median score
        median_scores = stuassess.groupby(['id_student'])['score'].median()\
        .reset_index(name='median_score')
        # merging median_scores to sv_si
        sv_si = sv_si.merge(median_scores, on='id_student')
        # merging assess & student_assessment data
        sv_si_sa = stuassess.merge(sv_si, on='id_student')
        # dropping dupe columns before merge
        drops = ['code_module', 'code_presentation', 'date', 'assessment_type']
        assess_df = assess_df.drop(drops, axis=1)
        # merging assess & student_assessment data
        sv_si_sa = sv_si_sa.merge(assess_df, on='id_assessment')
        # dropping extraneous columns before merge
        drops = ['code_module', 'code_presentation', 'id_student']
        sv_si_sa = sv_si_sa.drop(drops, axis=1)
        # decimalizing weight
        sv_si_sa['weight'] = sv_si_sa['weight'].apply(pd.to_numeric)*.01
        # getting adjusted scores
        sv_si_sa['adj_score'] = sv_si_sa['score']*sv_si_sa['weight']
        # weighted_ave df
        weighted_ave = sv_si_sa.groupby('row_id')['adj_score']\
        .sum().reset_index().rename(columns={'adj_score': 'weighted_ave'})
        # merging dfs
        sv_si = sv_si.merge(weighted_ave, on='row_id')
        # applying fixes
        sv_si = self.sql_fixes(sv_si)
        sv_si = self.sv_fixes(sv_si)
        return sv_si
    
    def pipe_cleaner_wa(self, df, col_list):
        def drop_cols(df, col_list):
            df = df.drop(col_list, axis=1)
            return df
        def null_zap(df):
            df = df.dropna()
            return df
        def replace_vals(df):
            # fixing typo
            df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
            # renaming values
            df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
            df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
            return df
        def drop_outliers_sc(df):
            Q1 = df.studied_credits.quantile(0.25)
            Q3 = df.studied_credits.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.studied_credits < (Q1 - 1.5 * IQR))\
                                  |(df.studied_credits > (Q3 + 1.5 * IQR)))].copy()
            return df
        def drop_outliers_wa(df):
            Q1 = df.weighted_ave.quantile(0.25)
            Q3 = df.weighted_ave.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.weighted_ave < (Q1 - 1.5 * IQR))\
                                  |(df.weighted_ave > (Q3 + 1.5 * IQR)))].copy()
            return df
        df = (df.pipe(replace_vals).pipe(null_zap)
              .pipe(drop_outliers_sc).pipe(drop_outliers_wa)
              .pipe(drop_cols, col_list))
        return df

    def sv_si_flipped(self):
        """
        Making df joining studentinfo & studentvle and creating a click_sum column.
        """
    
        q = """
        SELECT 
        SV.*,
        SI.*,
        /* creating the row_id column by concatenation*/
        SV.code_module || SV.code_presentation || SV.id_student AS row_id,
        /* creating the click_sum column*/
        SUM(SV.sum_click) AS click_sum,
        /* creating the num_activities column*/
        COUNT(SV.sum_click) AS num_activities,        
        /* creating binarized target column*/
        iif(SI.final_result='Pass' OR SI.final_result='Distinction', 0, 1) AS target
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
        # dropping duplicate columns
        df = df.T.drop_duplicates().T
        df = self.sql_fixes(df)
        df = self.sv_fixes(df)

        return df
    
    def df_a_flipped(self):
        # creating SVLE & SINFO df
        sv_si = self.sv_si_flipped()
        #creating ASSESSMENT df
        assess_df = self.simple_df('ASSESSMENTS')
        #creating STUDENTASSESSMENT df
        stuassess = self.simple_df('STUDENTASSESSMENT')
        # converting score DType
        stuassess['score'] = stuassess['score'].apply(pd.to_numeric)
        # getting each student's mean score
        mean_scores = stuassess.groupby(['id_student'])['score'].mean()\
        .reset_index(name='mean_score')
        #merging the dfs
        sv_si = sv_si.merge(mean_scores, on='id_student')
        # getting each student's median score
        median_scores = stuassess.groupby(['id_student'])['score'].median()\
        .reset_index(name='median_score')
        # merging median_scores to sv_si
        sv_si = sv_si.merge(median_scores, on='id_student')
        # merging assess & student_assessment data
        sv_si_sa = stuassess.merge(sv_si, on='id_student')
        # dropping dupe columns before merge
        drops = ['code_module', 'code_presentation', 'date', 'assessment_type']
        assess_df = assess_df.drop(drops, axis=1)
        # merging assess & student_assessment data
        sv_si_sa = sv_si_sa.merge(assess_df, on='id_assessment')
        # dropping extraneous columns before merge
        drops = ['code_module', 'code_presentation', 'id_student']
        sv_si_sa = sv_si_sa.drop(drops, axis=1)
        # decimalizing weight
        sv_si_sa['weight'] = sv_si_sa['weight'].apply(pd.to_numeric)*.01
        # getting adjusted scores
        sv_si_sa['adj_score'] = sv_si_sa['score']*sv_si_sa['weight']
        # weighted_ave df
        weighted_ave = sv_si_sa.groupby('row_id')['adj_score']\
        .sum().reset_index().rename(columns={'adj_score': 'weighted_ave'})
        # merging dfs
        sv_si = sv_si.merge(weighted_ave, on='row_id')
        # applying fixes
        sv_si = self.sql_fixes(sv_si)
        sv_si = self.sv_fixes(sv_si)
        return sv_si
