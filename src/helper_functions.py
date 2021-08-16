# importing 
import os, sys
import pickle

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, recall_score, precision_score,
                             make_scorer, plot_confusion_matrix, confusion_matrix)
from sklearn.inspection import permutation_importance

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
                          ['Satisfactory', 'Unsatisfactory'], ax=ax)
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
    precision = precision_score(y, model.predict(X))
    report = pd.DataFrame([[accuracy, f1, recall, precision]],\
                          columns=['Accuracy', 'F1', 'Recall', 'Precision'])    
    y_prob = model.predict(X)
    y_preds = np.where(y_prob > .5, 1,0) 
    c_matrix = confusion_matrix(y, y_preds)
    fig, ax = plt.subplots(figsize=(7,7))
    sns.heatmap(c_matrix, annot=True, fmt='d', cmap='GnBu_r',
               xticklabels=['Satisfactory', 'Unsatisfactory'], 
               yticklabels=['Satisfactory', 'Unsatisfactory'])
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
    df['course_load'] = df['course_load'].astype(str)
    # dropping extraneous column
    df = df.drop(columns=['sum_click'])
    
    return binarize_target(df)

def chi_sq_test(cross_tabs):
    chi2, p, dof, con_table = stats.chi2_contingency(cross_tabs)
    print(f'chi-squared = {chi2}\np value= {p}\ndegrees of freedom = {dof}')

def cramers_v(cross_tabs):
    """Returns the Cramer's V values for the various categories.
    
    Args:
        cross_tabs: A crosstab dataframe.
    Returns:
        Crarmer's V values for the various categories."""
    
    # getting the chi sq. stat
    chi2 = stats.chi2_contingency(cross_tabs)[0]
    # summing each category
    n = cross_tabs.sum()
    # getting the degrees of freedom
    dof = min(cross_tabs.shape)-1
    # returning cramer's v
    v = np.sqrt(chi2/(n*dof))
    # printing results
    print(f'Cramer\'s V Degrees of Freedom = {dof}\n')
    # returning cramer's v
    return v

def perm_importances(clf, X, y, scoring):
    """Plots the permution importances of the features.
    
    Args:
        clf: A classifier.
        X: A dataframe.
        y: A series.
        scoring: A string indicating the scoring object to be used.
    """
    importances = permutation_importance(clf, X, y, n_repeats=10,
                                         scoring=scoring,
                                         random_state=2021, n_jobs=-1)
    sorted_index = importances.importances_mean.argsort()
    data=importances.importances[sorted_index].T
    columns = X.columns[sorted_index]
    pi_df = pd.DataFrame(data=data, columns=columns)
    cols = [col.replace('_', ' ').title() for col in pi_df.columns]
    pi_df.columns = cols
    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(data=pi_df, orient='h', palette='GnBu_r')
    plt.show()

def cohens_d(sample1, sample2):
    """
    Returns Cohen's d value.
    
    Args: 
        sample1: A pandas series or numpy array.
        sample2: A pandas series of numpy array.
    Returns:
        Cohen's d value.
    """
    
    diff = sample1.mean() - sample2.mean()
    n1 = len(sample1)
    n2 = len(sample2)
    var1 = sample1.var(ddof=1)
    var2 = sample2.var(ddof=1)
    pooled_var = ((n1-1) * var1 + (n2-1) * var2) / (n1 + n2 - 2)
    d = diff / np.sqrt(pooled_var)
    return d