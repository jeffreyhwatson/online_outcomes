# just a placeholder file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import functions as fn

def outcomes_type(df):
    res = df.final_result.value_counts(normalize=True)\
    .reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='index',
                data=res, palette='winter_r')
    plt.title('Share of Outcomes by Type', fontsize=25)
    plt.ylabel('', fontsize=20)
    plt.xlabel('Percentage of Total Outcomes', fontsize=20)
    # plt.savefig('outcomes_type',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def outcomes_imd(df):
    imd = df.groupby('imd_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=imd,  hue='imd_band', palette='winter_r')
    plt.title('Share of Outcomes by IMD Band', fontsize=25)
    plt.ylabel('')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='IMD Band', bbox_to_anchor= (1, 1))
#     plt.savefig('outcomes_imd',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()

def outcomes_dis(df):
    dis = df.groupby('disability')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=dis,  hue='disability', palette='winter_r')
    plt.title('Share of Outcomes by Disability Status', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Disability', bbox_to_anchor= (1, 1))
    # plt.savefig('outcomes_dis',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    

def outcomes_age(df):
    age = df.groupby('age_band')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=age,  hue='age_band', palette='winter_r')
    plt.title('Share of Outcomes by Age Band', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Age Band', bbox_to_anchor= (1, 1))
#     plt.savefig('outcomes_age',  bbox_inches ="tight",\
#     pad_inches = .25, transparent = False)
    plt.show()

def outcomes_edu(df):
    edu = df.groupby('highest_education')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=edu,  hue='highest_education', palette='winter_r')
    plt.title('Share of Outcomes by Education Level', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Education Level', bbox_to_anchor= (1, 1))
#     plt.savefig('outcomes_edu',  bbox_inches ="tight",\
#     pad_inches = .25, transparent = False)
    plt.show()

def outcomes_gen(df):
    gen = df.groupby('gender')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=gen,  hue='gender', palette='winter_r')
    plt.title('Share of Outcomes by Gender', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Gender', bbox_to_anchor= (1, 1))
    # plt.savefig('outcomes_gen',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def outcomes_reg(df):
    reg = df.groupby('region')['final_result']\
    .value_counts(normalize=True).reset_index(name='percentage')

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='percentage', y ='final_result',
                data=reg,  hue='region', palette='winter_r')
    plt.title('Share of Outcomes by Region', fontsize=25)
    plt.ylabel(' ')
    plt.xlabel('Percentage of Outcome', fontsize=20)
    plt.legend(title='Region', bbox_to_anchor= (1, 1))
    # plt.savefig('outcomes_reg',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def base_coefs(pipe):
    coefs = pipe[1].coef_.flatten()
    features = pipe[0].get_feature_names()
    features = fn.feat_cleaner(features)
    zips = zip(features, coefs)
    coef_df = pd.DataFrame(zips, columns=['feature', 'value'])
    coef_df["abs_value"] = coef_df["value"].apply(lambda x: abs(x))
    coef_df["colors"] = coef_df["value"].apply(lambda x: "darkblue" if x > 0 else "skyblue")
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
#     plt.savefig('tuned_coeff',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()
    
def base_neg_odds(pipe):
    coefs = pipe[1].coef_.flatten()
    features = pipe[0].get_feature_names()
    features = fn.feat_cleaner(features)

    odds = np.exp(coefs)
    odds_df = pd.DataFrame(odds, 
                 features, 
                 columns=['odds'])\
                .sort_values(by='odds', ascending=False)

    top10_neg_odds = odds_df.head(10).reset_index()

    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(x='index',y='odds', data=top10_neg_odds, palette='Blues_r',
                edgecolor='deepskyblue')
    plt.suptitle('Relative Odds For The Top 10 Negative Features', fontsize=30)
    plt.title('Higher Bars Mean Higher Odds of a Negative Tweet', fontsize=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=80)
#     plt.savefig('tuned_negative',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()

    
def base_pos_odds(pipe):
    coefs = pipe[1].coef_.flatten()
    features = pipe[0].get_feature_names()
    features = fn.feat_cleaner(features)

    odds = np.exp(coefs)
    odds_df = pd.DataFrame(odds, 
                 features, 
                 columns=['odds'])\
                .sort_values(by='odds', ascending=True)
    
    top10_pos_odds = odds_df.head(10).reset_index()

    top10_pos_odds['odds'] = 1/top10_pos_odds['odds']

    fig, ax = plt.subplots(figsize =(20, 8))
    sns.barplot(x='index',y='odds', data=top10_pos_odds, palette='Blues_r', edgecolor='deepskyblue')
    plt.suptitle('Relative Odds For Top 10 Positive Features', fontsize=30)
    plt.title('Higher Bars Mean Higher Odds of a Positive Tweet', fontsize=20)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks(rotation=80)
#     plt.savefig('tuned_positive',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()
    
def feature_plot(pipe):
    """Returns feature importances of a model."""
    
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
#     plt.savefig('feature_imp',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()