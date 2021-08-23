import os, sys

# setting project path
gparent = os.path.join(os.pardir, os.pardir)
sys.path.append(gparent)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import helper_functions as fn

def outcomes_target(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    res = df.target.value_counts(normalize=True)\
    .reset_index(name='Percentage')
    res = res.rename(columns={'index': 'Outcome'})
    res['Outcome'] = res['Outcome'].apply(lambda x: \
                                          'Satisfactory' if x==0 else 'Unsatisfactory')
    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Outcome', y ='Percentage',
                data=res, palette='GnBu_r', edgecolor='lightseagreen')
    plt.title('Share of Outcomes', fontsize=25)
    plt.ylabel('', fontsize=20)
    plt.xlabel('Percentage of Total Outcomes', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    return res

def outcomes_type(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    res = df.final_result.value_counts(normalize=True)\
    .reset_index(name='Percentage')
    res = res.rename(columns={'index': 'Outcome'})
    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Percentage', y ='Outcome',
                data=res, palette='GnBu_r', edgecolor='lightseagreen')
    plt.title('Share of Outcomes by Type', fontsize=25)
    plt.ylabel('', fontsize=20)
    plt.xlabel('Percentage of Total Outcomes', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    return res
    
def outcomes_imd(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    imd = df.groupby('imd_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='Percentage')
    imd = imd.rename(columns={'index': 'Outcome'})
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='imd_band', y ='Percentage',
                data=imd,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes By IMD Band', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('IMD Band', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_dis(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    dis = df.groupby('disability')['final_result']\
    .value_counts(normalize=True).reset_index(name='Percentage')
    dis = dis.rename(columns={'disability':	'Disability', 'final_result': 'Outcome'})
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='Disability', y ='Percentage',
                data=dis,  hue='Outcome', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes By Disability Status', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xticks([0,1], labels=['Nondisabled', 'Disabled'])
    plt.xlabel('')
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    return dis

def outcomes_age(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    age = df.groupby('age_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='age_band', y ='percentage',
                data=age,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Age Band', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Age Band', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_edu(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    edu = df.groupby('highest_education')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='highest_education', y ='percentage',
                data=edu,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes By Education Level', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('')
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
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
    sns.barplot(x='gender', y ='percentage',
                data=gen,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Gender', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('')
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_att(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    att = df.groupby('num_of_prev_attempts')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='num_of_prev_attempts', y ='percentage',
                data=att,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Previous Attempts', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Number of Attempts', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()    

def outcomes_mod(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    att = df.groupby('code_module')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='code_module', y ='percentage',
                data=att,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Module', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Module Code', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()   

def outcomes_pres(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    att = df.groupby('code_presentation')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='code_presentation', y ='percentage',
                data=att,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Presentation', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Presentation Code', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
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
    sns.barplot(x='region', y ='percentage',
                data=reg,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Region', fontsize=25)
    plt.ylabel('Percentage')
    plt.xlabel('', fontsize=20)
    plt.xticks(rotation=60)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
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
    sns.barplot(x='course_load', y='percentage', hue='final_result',\
                data=stu, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Course Load', fontsize=25)
    plt.xlabel('')
    plt.ylabel('Percentage', fontsize=20)
    plt.legend(title="Outcome")
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcome_clks(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    clix = df.groupby('final_result')['click_sum']\
    .mean().reset_index(name='mean_clicks')
    
    fig, ax=plt.subplots(figsize=(16,8))
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
    
    fig, ax=plt.subplots(figsize=(16,8))
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

    
def outcomes_wa(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    mwa = df.groupby('final_result')['weighted_ave'].mean()\
    .reset_index(name='mean_wa').round()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='mean_wa', y ='final_result',
                data=mwa, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Outcome by Assessment Weighted Average', fontsize=25)
    plt.ylabel('')
    plt.xlabel('Mean Weighted Average of Assessment Scores', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def outcomes_med(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    mms = df.groupby('final_result')['median_score'].mean()\
    .reset_index(name='mean_ms').round()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='mean_ms', y ='final_result',
                data=mms, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Outcome by Median Assessment Score', fontsize=25)
    plt.ylabel('')
    plt.xlabel('Average of Median Assessment Scores', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_mean(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    mes = df.groupby('final_result')['mean_score'].mean()\
    .reset_index(name='mean_score').round()
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='mean_score', y ='final_result',
                data=mes, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Outcome by Mean Assessment Score', fontsize=25)
    plt.ylabel('')
    plt.xlabel('Average of Mean Assessment Score', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()

def outcomes_qc(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    q1a = df.groupby('final_result')['q1_clicks']\
    .value_counts(normalize=True).reset_index(name='values')
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='q1_clicks', y ='values', hue='final_result',
                data=q1a, palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes by Q1 Click Levels', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('')
    plt.legend(title = 'Outcome')
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)    
    plt.show()    

def outcomes_sumact(df, plot_name=False):
    """If a plot_name string is provided then a figure is saved to the figure directory."""
    
    act = df.groupby('activity_level')['final_result']\
    .value_counts(normalize=True).reset_index(name='Percentage')
    act = act.rename(columns={'index': 'Outcome'})
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='activity_level', y ='Percentage',
                data=act,  hue='final_result', palette='GnBu_r',
                edgecolor='lightseagreen')
    plt.title('Percentage of Outcomes By Activity Level', fontsize=25)
    plt.ylabel('Percentage', fontsize=20)
    plt.xlabel('Activity Level', fontsize=20)
    plt.legend(title='Outcome', bbox_to_anchor= (1, 1))
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)    
    
def importance_plot(pipeline, X, plot_name=False):
    """Returns feature importances of a classifier."""
    
    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)   
    importances = pipeline[1].feature_importances_
    sorted_importances = sorted(list(zip(features, importances)),\
                                key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    sns.barplot(x=x, y=y, palette='GnBu_r', edgecolor='lightseagreen')
    plt.xticks(rotation=80)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def importance_plot_bclf(pipeline, X, plot_name=False):
    """Returns feature importances of a classifier."""
    
    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)   
    importances = np.mean([tree.feature_importances_\
                           for tree in pipeline[2].estimators_], axis=0)
    sorted_importances = sorted(list(zip(features, importances)),\
                                key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    sns.barplot(x=x, y=y, palette='GnBu_r', edgecolor='lightseagreen')
    plt.xticks(rotation=80)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()    
    
def importance_plot_sm(pipeline, X, plot_name=False):
    """Returns feature importances of a classifier."""
    
    features = list(pipeline[0].transformers_[0][1].get_feature_names()) +\
                list(X.select_dtypes('number').columns)    
    importances = pipeline[2].feature_importances_
    sorted_importances = sorted(list(zip(features, importances)),\
                                key=lambda x: x[1], reverse=True)[:25]
    x = [val[0] for val in sorted_importances]
    y = [val[1] for val in sorted_importances]
    plt.figure(figsize=(20,6))
    sns.barplot(x=x, y=y, palette='GnBu_r', edgecolor='lightseagreen')
    plt.xticks(rotation=80)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show() 
    
def error_rate(df, plot_name=False):
    """"Plots the error rate for each of the labels."""
    
    label_counts = df.label.value_counts()
    label_errors = df[df['result'] != df['prediction']]['label'].value_counts()
    error_rates = label_errors/label_counts
    er_df = pd.DataFrame(error_rates).reset_index()
    fig, ax=plt.subplots(figsize=(16,8))
    sns.barplot(x='index', y='label', data=er_df)
    plt.title('Error Rate By Outcome', fontsize=25)
    plt.xticks(fontsize=20)
    plt.xlabel('')
    plt.ylabel('Error Rate', fontsize=20)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def numerical_errors(df, num_cols, plot_name=False):
    """"Plots the mean values of features across outcome & prediction type."""
    
    num_cols = num_cols
    num_errors = df[num_cols]
    fig, ax = plt.subplots(5,1, figsize = (20,40))
    ax = ax.ravel()
    for idx, col in enumerate(num_cols):
        data = pd.DataFrame()
        data['All Students'] = df[col]
        data['Students That Passed'] = df[df['label'] == 'Pass'][col]
        data['False Pass Predictions'] = df[(df['result']
                                                 != df['prediction'])
                                                & (df['label'] == 'Pass')][col]    
        data['True Pass Predictions'] = df[(df['result'] 
                                                == df['prediction'])
                                               & (df['label'] == 'Pass')][col]  
        data['Students That Failed'] = df[df['label'] == 'Fail'][col]
        data['False Fail Predictions'] = df[(df['result'] 
                                                 != df['prediction'])
                                                & (df['label'] == 'Fail')][col]    
        data['True Fail Predictions'] = df[(df['result'] 
                                                 == df['prediction'])
                                                & (df['label'] == 'Fail')][col] 
        sns.barplot(data=data, ax=ax[idx], palette='GnBu_r', 
                    edgecolor='lightseagreen')
        ax[idx].set(title=f"Average {col.replace('_', ' ').title()}")
    fig.tight_layout(pad = 7, h_pad = 3, w_pad = 5)
    plt.suptitle('Mean Feature Values Across Outcome And Prediction Type',
                 fontsize=30 )
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()
    
def categorical_errors(df, col, rotation=False, plot_name=False):   
    fig, ax = plt.subplots(figsize = (16,8))
    sns.barplot(data=df, x=col, y='true_prediction', palette='GnBu_r', edgecolor='lightseagreen')
    plt.title(f"Percent Predicted Correctly By {col.replace('_', ' ').title()}")
    plt.xlabel('')
    plt.ylabel('Pecent Correct', fontsize=20)
    if rotation !=False:
        plt.xticks(rotation=rotation)
    path = os.path.join(gparent,'reports/figures',f'{plot_name}.png')
    if plot_name!=False:
        plt.savefig(path,  bbox_inches ="tight",\
                    pad_inches = .25, transparent = False)
    plt.show()