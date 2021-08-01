# importing 
import os, sys
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             make_scorer, plot_confusion_matrix, confusion_matrix)

import matplotlib.pyplot as plt
import matplotlib.image as img
import dataframe_image as dfi
import seaborn as sns

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

def fetch(cur, q):
    """Returns an SQL query."""
    
    return cur.execute(q).fetchall()

def acc_score(y_true, y_pred):
    "Accuracy scoring function for use in make_scorer."
    
    acc = accuracy_score(y_true, y_pred)
    return acc

# creating scorer object
acc = make_scorer(acc_score)

def f_score(y_true, y_pred):
    "F1 scoring function for use in make_scorer."
    
    f1 = f1_score(y_true, y_pred)
    return f1

# creating scorer object
f1 = make_scorer(f_score)

def recall(y_true, y_pred):
    "Recall scoring function for use in make_scorer."
    
    rec = recall_score(y_true, y_pred)
    return rec

# creating scorer object
recall = make_scorer(recall)

def precision(y_true, y_pred):
    "Precision scoring function for use in make_scorer."
    
    pre = precision_score(y_true, y_pred)
    return pre

# creating scorer object
precision = make_scorer(precision)

def X_y(df):
    """Returns a data frame and target series."""
    
    X = df.drop('target', axis=1)
    y = df['target']
    return X, y

def test_train(X, y):
    """Returns a train/test split."""
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        random_state=2021,
                                                        stratify=y
                                                   )
    return  X_train, X_test, y_train, y_test

def confusion_report(model, X, y, plot_name=False):
    """
    Returns a confusion matrix plot and scores.
    
    If a plot_name string is provided then a figure is saved to the figure directory.
    """
    
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    accuracy = accuracy_score(y, model.predict(X))
    f1 = f1_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    precision = precision_score(y, model.predict(X),)
    report = pd.DataFrame([[accuracy, f1, recall, precision]],\
                          columns=['Accuracy', 'F1', 'Recall', 'Precision']) 
    
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_confusion_matrix(model, X, y,
                          cmap='GnBu_r',
                          display_labels=
                          ['Unsatisfactory', 'Satisfactory'], ax=ax)
    plt.title('Confusion Matrix')
    plt.grid(False)
    if plot_name != False:
        plt.savefig(path,  bbox_inches ="tight",\
                pad_inches = .25, transparent = False)
    plt.show()  
    
    return report

def confusion_report_nn(model, X, y, plot_name=False):
    """
    Returns a confusion matrix plot and scores.
    
    If a plot_name string is provided then a figure is saved to the figure directory.
    """
    
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    accuracy = accuracy_score(y, model.predict(X))
    f1 = f1_score(y, model.predict(X))
    recall = recall_score(y, model.predict(X))
    precision = precision_score(y, model.predict(X),)
    report = pd.DataFrame([[accuracy, f1, recall, precision]],\
                          columns=['Accuracy', 'F1', 'Recall', 'Precision'])    
    y_prob = model.predict(X)
    y_preds = np.where(y_prob > .5, 1,0) 
    c_matrix = confusion_matrix(y, y_preds)
    fig, ax = plt.subplots(figsize=(7,7))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='GnBu_r',
               xticklabels=['Unsatisfactory', 'Satisfactory'], 
               yticklabels=['Unsatisfactory', 'Satisfactory'])
    plt.title('Confusion Matrix')
    plt.yticks(rotation=0)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if plot_name != False:
        plt.savefig(path,  bbox_inches ="tight",\
            pad_inches = .25, transparent = False)    
    plt.show()
    return report

def df_plot(df, plot_name=False):
    """Saves a plot of a data frame to the figure directory."""
    
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    dfi.export(df,f'{path}', max_rows=-1)

def pickle_model(model, model_name):
    """Pickles model and save to model directory."""
    
    path = os.path.join(gparent, 'models', f'{model_name}.pkl')
    file = open(path, 'wb')
    pickle.dump(model, file)
    file.close()

def col_pop(df, column, index=False):
    """Moves column to index given"""
    
    if index == False:
        index = 0
    else:
        index=index
    col_name = column
    col = df.pop(col_name)
    return df.insert(index ,col_name, col)    
    
def binarize_target(df):
    """Binarizes target and moves column to front of data frame."""
    
    binarize = {'Pass': 1, 'Withdrawn': 0, 'Fail': 0, 'Distinction': 1}
    df['target'] = df['final_result']
    df['target'] = df['target'].map(binarize)
    col_name = 'target'
    first = df.pop(col_name)
    df.insert(0, col_name, first)
    return df

def sv_si_fixes(df):
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
    conversions = ['click_sum','num_activities','date',
                   'num_of_prev_attempts','studied_credits']
    df[conversions] = df[conversions].apply(pd.to_numeric)
    # adding course_load column
    df['course_load'] = pd.qcut(df.studied_credits, q=4,\
                                  labels=['Light', 'Medium', 'Heavy'],\
                                  duplicates='drop')
    # dropping extraneous column
    df = df.drop(columns=['sum_click'])
    
    return binarize_target(df)