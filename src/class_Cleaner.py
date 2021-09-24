import pandas as pd
import numpy as np
import os, sys
import sqlite3
from src import helper_functions as f

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

class Cleaner:
    
#    init method
    
    def __init__(self, database_name):
        """Creates database, cursor, and connects to database."""
         
        data_path = os.path.join(gparent,'data/processed','outcomes.db')
        self.conn = sqlite3.connect(data_path)  
        self.cur = self.conn.cursor()

#   cleaning methods
    def simple_df(self, table_name):
        """Returns data frame made from all entries in table."""
        
        q = f"SELECT*FROM {table_name}"
        df = pd.read_sql(q, self.conn)
        return df
        
    def sql_fixes(self, df):
        """Performs various fixes to the dataframe."""
    
        # moving target & row_id to front of df
        f.col_pop(df, 'target')
        f.col_pop(df, 'row_id', 1)
        # converting data type
        converts = ['num_of_prev_attempts', 'studied_credits']
        df[converts] = df[converts].apply(pd.to_numeric)
        return df

    def student_info_full(self):
        """Returns a dataframe of STUDENTINFO data."""
            
        # creating student info df
        q="""
        SELECT*,
        /*creating row_id col*/
        code_module || code_presentation || id_student AS row_id,
        /* creating binarized target column*/
        IIF(final_result='Pass' OR final_result='Distinction', 0, 1) AS target      
        FROM STUDENTINFO"""
        df = pd.read_sql(q, self.conn)
           
        return self.sql_fixes(df)

    def student_info_vle_full(self):
        """Returns a dataframe of STUDENTINFO & STUDENTVLE data."""
            
        # creating studentvle df
        q = """
        SELECT SV.*,
        SI.*,
        /*creating click sum*/
        SUM(SV.sum_click) AS click_sum,
        /*creating num_activities*/
        COUNT(SV.sum_click) AS num_activities,
        /*creating row_id*/
        SV.code_module || SV.code_presentation || SV.id_student AS row_id,
        /* creating binarized target column*/
        IIF(final_result='Pass' OR final_result='Distinction', 0, 1) AS target          
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
           
        return self.sql_fixes(df)

    def student_info_assessment_full(self):
        """returns a dataframe of STUDENTINFO & STUDENTASSESSMENT data."""
        
        q = f"""
        SELECT
        /*selecting all from student info*/
        SI.*,
        /*changing dtype*/
        CAST(SA.date_submitted AS INTEGER) AS date_sub,
        /*adding course length*/
        CAST(C.module_presentation_length AS INTEGER) AS course_length,
        /* creating the row_id column by concatenation*/
        SI.code_module || SI.code_presentation || SI.id_student AS row_id,
        /* creating binarized target column*/
        IIF(SI.final_result='Pass' OR SI.final_result='Distinction', 0, 1) AS target,
        /* creating weighted_ave column*/
        SUM(SA.score*A.weight*.01) AS weighted_ave,
        /* creating mean_score column*/
        AVG(SA.score) as mean_score
        FROM STUDENTASSESSMENT AS SA
        JOIN
        ASSESSMENTS AS A
        ON A.id_assessment = SA.id_assessment
        JOIN STUDENTINFO AS SI
        ON SI.id_student = SA.id_student
        JOIN COURSES AS C
        ON SI.code_module = C.code_module
        GROUP BY SA.id_student,
        SI.code_module,
        SI.code_presentation;
        """
        df = pd.read_sql(q, self.conn)
        return self.sql_fixes(df)
    
    def studentvle_full(self):
        """Returns a dataframe of STUDENTVLE data."""
        
        q=f"""
        SELECT
        /*selecting all from STUDENTVLE*/
        SV.*,
        /* creating the row_id column by concatenation*/
        SV.code_module || SV.code_presentation || SV.id_student AS row_id,
        /* creating the sum_activity column*/
        SUM(SV.sum_click) + COUNT(SV.sum_click) AS sum_activity,
        CAST(SV.date AS INTEGER) AS max_date
        FROM
        STUDENTVLE AS SV
        GROUP BY 
        SV.code_module,
        SV.code_presentation,
        SV.id_student;
        """
        df = pd.read_sql(q, self.conn)
        return df
    
    def median_score_full(self):
        """Returns a dataframe of median assessment scores."""
            
        # creating registration data df
        df = self.simple_df('STUDENTASSESSMENT')
        # converting datatypes
        converts = ['date_submitted', 'score']
        df[converts] = df[converts].apply(pd.to_numeric) 
        # creating median_score df
        df = df.groupby(['id_student'])['score'].median().reset_index(name='median_score')           
        return df     

    def data_prep_full(self):
        """Returns a data frame created from several tables."""
        
        median_score = self.median_score_full()
        student_vle = self.studentvle_full()
        df = self.student_info_assessment_full()
        
        df = df.merge(median_score, on='id_student')
        drops = ['code_module','code_presentation','id_student']
        df = df.drop(drops, axis=1)
        drops = ['code_module','code_presentation','id_student']
        student_vle = student_vle.drop(drops, axis=1)
        df = df.merge(student_vle, on='row_id')
        return self.sql_fixes(df)

    def cleaner_studentinfo(self, df, col_list):
        """Returns a cleaned dataframe."""

        def replace_vals(df):
            # fixing typo
            df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
            # renaming values
            df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
            df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
            return df
        def drop_outliers_sc(df):
            # dropping studied_credits outliers with IQR fences
            Q1 = df.studied_credits.quantile(0.25)
            Q3 = df.studied_credits.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.studied_credits < (Q1 - 1.5 * IQR))\
                                  |(df.studied_credits > (Q3 + 1.5 * IQR)))].copy()
            return df
        def course_load(df):
            # binning studied_credits data and creating a course_load column
            df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                labels=['Light', 'Medium', 'Heavy'],\
                                duplicates='drop')
            return df
        def drop_cols(df, col_list):
            # dropping columns
            df = df.drop(col_list, axis=1)
            return df
        def null_zap(df):
            # dropping nulls
            df = df.dropna()
            return df  
        # applying the cleaning functions
        df = (df.pipe(replace_vals)
              .pipe(drop_outliers_sc).pipe(course_load)
              .pipe(drop_cols, col_list)).pipe(null_zap)
        return df    
    
    
    def registration_data_df(self, cutoff_date):
        """Returns a dataframe of data with either no withdrawl date or a date after the cutoff parameter."""
            
        # creating registration data df
        q="""
        SELECT date_registration, date_unregistration,
        /*creating row_id col*/
        code_module || code_presentation || id_student AS row_id
        FROM STUDENTREGISTRATION"""
        df = pd.read_sql(q, self.conn)
        # converting datatypes
        converts = ['date_registration', 'date_unregistration']
        df[converts] = df[converts].apply(pd.to_numeric)  
        # filtering out withdrawls before the cutoff_date
        df = df[(df.date_unregistration.isna())|\
                (df.date_unregistration > cutoff_date)]
        # filtering row_ids with no registration data
        df = df[~df.date_registration.isna()]            
        return df
    
    def median_score_df(self, cutoff_date):
        """Returns a dataframe of data of median assessment scores upto the cutoff parameter."""
            
        # creating registration data df
        df = self.simple_df('STUDENTASSESSMENT')
        # converting datatypes
        converts = ['date_submitted', 'score']
        df[converts] = df[converts].apply(pd.to_numeric) 
        # filtering out submissions after cutoff_date
        df = df[df.date_submitted < cutoff_date]
        # creating median_score df
        df = df.groupby(['id_student'])['score'].median().reset_index(name='median_score')           
        return df
    
    def student_info_assessment_df(self, cutoff_date):
        """returns a dataframe of student info & assessment data upto the cutoff_date"""
        
        q = f"""
        SELECT
        /*selecting score*/
        SA.score,
        /*selecting all from student info*/
        SI.*,
        /*changing dtype*/
        CAST(SA.date_submitted AS INTEGER) AS date_sub,
        /*adding course length*/
        CAST(C.module_presentation_length AS INTEGER) AS course_length,
        /* creating the row_id column by concatenation*/
        SI.code_module || SI.code_presentation || SI.id_student AS row_id,
        /* creating binarized target column*/
        IIF(SI.final_result='Pass' OR SI.final_result='Distinction', 0, 1) AS target,
        /* creating weighted_ave column*/
        SUM(SA.score*A.weight*.01) AS weighted_ave,
        /* creating mean_score column*/
        AVG(SA.score) as mean_score
        FROM STUDENTASSESSMENT AS SA
        JOIN
        ASSESSMENTS AS A
        ON A.id_assessment = SA.id_assessment
        JOIN STUDENTINFO AS SI
        ON SI.id_student = SA.id_student
        JOIN COURSES AS C
        ON SI.code_module = C.code_module
        WHERE date_sub < {cutoff_date}
        GROUP BY SA.id_student,
        SI.code_module,
        SI.code_presentation;
        """
        df = pd.read_sql(q, self.conn)
        return df
    
    def studentvle_df(self, cutoff_date):
        """Returns a dataframe of STUDENTVLE data upto the cutoff date."""
        
        q=f"""
        SELECT
        /*selecting all from STUDENTVLE*/
        SV.*,
        /* creating the row_id column by concatenation*/
        SV.code_module || SV.code_presentation || SV.id_student AS row_id,
        /* creating the sum_activity column*/
        SUM(SV.sum_click) + COUNT(SV.sum_click) AS sum_activity,
        CAST(SV.date AS INTEGER) AS max_date
        FROM
        STUDENTVLE AS SV
        WHERE
        max_date < {cutoff_date}
        GROUP BY 
        SV.code_module,
        SV.code_presentation,
        SV.id_student;
        """
        df = pd.read_sql(q, self.conn)
        return df

    def data_prep_half(self, cutoff_date):
        """Returns a data frame created from several tables."""
        
        reg_data = self.registration_data_df(cutoff_date)
        median_score = self.median_score_df(cutoff_date)
        student_vle = self.studentvle_df(cutoff_date)
        df = self.student_info_assessment_df(cutoff_date)
        
        df = df.merge(median_score, on='id_student')
        drops = ['code_module','code_presentation','id_student']
        df = df.drop(drops, axis=1)
        df = df.merge(student_vle, on='row_id')
        df = df.merge(reg_data, how='right', on='row_id')
        return self.sql_fixes(df)

    def pipe_cleaner(self, df, col_list):
        """Returns a cleaned dataframe."""
        
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
    
    def pipe_cleaner_wa(self, df, col_list):
        """Returns a cleaned dataframe."""

        def replace_vals(df):
            # fixing typo
            df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
            # renaming values
            df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
            df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
            return df
        def drop_outliers_sc(df):
            # dropping studied_credits outliers with IQR fences
            Q1 = df.studied_credits.quantile(0.25)
            Q3 = df.studied_credits.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.studied_credits < (Q1 - 1.5 * IQR))\
                                  |(df.studied_credits > (Q3 + 1.5 * IQR)))].copy()
            return df
        def course_load(df):
            # binning studied_credits data and creating a course_load column
            df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                labels=['Light', 'Medium', 'Heavy'],\
                                duplicates='drop')
            return df        
        def drop_outliers_wa(df):
            # dropping weighted_ave outliers with IQR fences
            Q1 = df.weighted_ave.quantile(0.25)
            Q3 = df.weighted_ave.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.weighted_ave < (Q1 - 1.5 * IQR))\
                                  |(df.weighted_ave > (Q3 + 1.5 * IQR)))].copy()
            return df
        def activity_level(df):
            # binning sum_activity data and creating a activity_level column
            df['activity_level'] = pd.qcut(df.sum_activity, q=4,\
                                labels=['Very Light','Light', 'Medium', 'Heavy'],\
                                duplicates='drop')
            return df        
        def drop_outliers_sa(df):
            Q1 = df.sum_activity.quantile(0.25)
            Q3 = df.sum_activity.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.sum_activity < (Q1 - 1.5 * IQR))\
                      |(df.sum_activity > (Q3 + 1.5 * IQR)))].copy()
            return df
        def drop_cols(df, col_list):
            # dropping columns
            df = df.drop(col_list, axis=1)
            return df
        def null_zap(df):
            # dropping nulls
            df = df.dropna()
            return df  
        # applying the cleaning functions
        df = (df.pipe(replace_vals).pipe(drop_outliers_sc)
              .pipe(drop_outliers_wa).pipe(drop_outliers_sa).pipe(activity_level)
              .pipe(course_load).pipe(drop_cols,col_list).pipe(null_zap))
        return df
    
    def cleaner_studentinfo(self, df, col_list):
        """Returns a cleaned dataframe."""

        def replace_vals(df):
            # fixing typo
            df['imd_band'] = df['imd_band'].replace(['10-20'], '10-20%')
            # renaming values
            df['disability'] = df['disability'].replace(['Y', 'N'], ['Yes', 'No'])
            df['gender'] = df['gender'].replace(['M', 'F'], ['Male', 'Female'])
            return df
        def drop_outliers_sc(df):
            # dropping studied_credits outliers with IQR fences
            Q1 = df.studied_credits.quantile(0.25)
            Q3 = df.studied_credits.quantile(0.75)
            IQR = Q3 - Q1
            df = df[~((df.studied_credits < (Q1 - 1.5 * IQR))\
                                  |(df.studied_credits > (Q3 + 1.5 * IQR)))].copy()
            return df
        def course_load(df):
            # binning studied_credits data and creating a course_load column
            df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                labels=['Light', 'Medium', 'Heavy'],\
                                duplicates='drop')
            return df
        def drop_cols(df, col_list):
            # dropping columns
            df = df.drop(col_list, axis=1)
            return df
        def null_zap(df):
            # dropping nulls
            df = df.dropna()
            return df  
        # applying the cleaning functions
        df = (df.pipe(replace_vals)
              .pipe(drop_outliers_sc).pipe(course_load)
              .pipe(drop_cols, col_list).pipe(null_zap))
        return df 