import os, sys

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import helper_functions as fn

def outcomes_type(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    res = df.final_result.value_counts(normalize=True)\
    .reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='index',
                data=res, palette='GnBu_r', edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Type', fontsize=25)
    plt.ylabel('', fontsize=20)
    plt.xlabel('Percentage of Total Outcomes', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def outcomes_imd(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    imd = df.groupby('imd_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=imd,  hue='imd_band', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by IMD Band', fontsize=25)
    plt.ylabel('')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='IMD Band', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def outcomes_dis(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    dis = df.groupby('disability')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=dis,  hue='disability', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Disability Status', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Disability', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    

def outcomes_age(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    age = df.groupby('age_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=age,  hue='age_band', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Age Band', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Age Band', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_edu(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    edu = df.groupby('highest_education')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=edu,  hue='highest_education', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Education Level', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Education Level', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_gen(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    gen = df.groupby('gender')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=gen,  hue='gender', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Gender', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Gender', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_reg(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    reg = df.groupby('region')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=reg,  hue='region', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Region', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Region', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcome_cl(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    stu = df.groupby('course_load')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')
    
    fig, ax=plt.subplots(figsize=(20,8))
    sns.barplot(x='percentage', y='final_result', hue='course_load',\
                data=stu, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Outcomes By Course Load', fontsize=25)
    plt.xlabel('Percentage Of Outcome', fontsize=20)
    plt.ylabel('')
    plt.legend(title="Course Load")
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcome_clks(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    clix = df.groupby('final_result')['click_sum']\
    .mean().reset_index(name='mean_clicks')
    
    fig, ax=plt.subplots(figsize=(20,8))
    sns.barplot(x='mean_clicks', y='final_result', data=clix, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Mean Clicks Per Outcome', fontsize=25)
    plt.xlabel('Mean Number Of Clicks', fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('')
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def outcome_nact(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    clix = df.groupby('final_result')['num_activities']\
    .mean().reset_index(name='mean_activities')
    
    fig, ax=plt.subplots(figsize=(20,8))
    sns.barplot(x='mean_activities', y='final_result', data=clix, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Mean Activities Per Outcome', fontsize=25)
    plt.xlabel('Mean Number Of Activities', fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('')
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def heat_map(corr, plot_name=False):
    """
    Returns a heatmap of a correlation matrix
    
    If a plot_name string is provided then a figure is saved to the figure directory.
    Args:
        corr: A corelation matrix object.
    """
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    fig1, ax1 = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap='viridis');
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
    
def base_coefs(pipe, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory. """
    
    coefs = pipe[1].coef_.flatten()
    features = pipe[0].get_feature_names()
    zips = zip(features, coefs)
    coef_df = pd.DataFrame(zips, columns=['feature', 'value'])
    coef_df["abs_value"] = coef_df["value"].apply(lambda x: abs(x))
    coef_df["colors"] = coef_df["value"].apply(lambda x: "darkblue" if x > 0 else "lightseagreen")
    coef_df = coef_df.sort_values("abs_value", ascending=False)

    fig, ax = plt.subplots(1, 1, figsize=(20, 8))
    sns.barplot(x="feature",
                y="value",
                data=coef_df.head(30),
               palette=coef_df.head(30)["colors"])
    ax.set_xticklabels(ax.get_xticklabels(), rotation=80, fontsize=20)
    ax.set_title("Dark Blue = Negative Sentiment Features", fontsize=20)
    plt.suptitle("Top 30 Features", fontsize=30)
    ax.set_ylabel("Coefs", fontsize=22)
    ax.set_xlabel("Feature Name", fontsize=22)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def feature_plot(pipe):
    """
    Returns feature importances of a model.
    
    If a plot_name string is provided then a figure is saved to the figure directory.
    """
    
    features = list(pipe[0].get_feature_names())
    features = fn.feat_cleaner(features)
    importances = pipe[1].feature_importances_
    sorted_importances = sorted(list(zip(features, importances)),
                                key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    
    plt.figure(figsize=(20,8))
    sns.barplot(x=x, y=y, palette='Blues_r', edgecolor='deepskyblue')
    plt.xticks(rotation=80, fontsize=20)
    plt.title('Feature Importances', fontsize=30)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

    