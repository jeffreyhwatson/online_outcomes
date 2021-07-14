# just a placeholder file

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src import functions as fn

def brand_freqs(df):
    brands = df.brand_product.value_counts(normalize=True)
    brands_df = pd.DataFrame(brands)
    brands_df.reset_index(inplace=True)
    brands_df.columns = ['Brand/Product', 'Percentage']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Percentage', y='Brand/Product', edgecolor='deepskyblue', palette='Blues_r', data=brands_df)
    ax.tick_params(labelsize=20)
    plt.title('Brand Share of Data', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('brand_freqs',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def emotion_freqs(df):
    emotion = df.emotion.value_counts(normalize=True)
    emotion_df = pd.DataFrame(emotion)
    emotion_df.reset_index(inplace=True)
    emotion_df.columns = ['Emotion', 'Share']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Share', y='Emotion', edgecolor='deepskyblue', palette='Blues_r', data=emotion_df)
    ax.tick_params(labelsize=20)
    plt.title('Emotion Share of Data', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
#     plt.savefig('aug_emotion_share',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
    plt.show()

def null_brand_emotions(df):
    null_brand_emotion = df[(df['brand_product'].isna()) &\
     (df['emotion'] != 'No emotion toward brand or product' )]
    emotion = null_brand_emotion.emotion.value_counts(normalize=True)
    emotion_df = pd.DataFrame(emotion)
    emotion_df.reset_index(inplace=True)
    emotion_df.columns = ['Emotion', 'Count']

    fig, ax = plt.subplots(figsize=(20,8))
    sns.barplot(x='Count', y='Emotion', edgecolor='deepskyblue',
                palette='Blues_r', data=emotion_df)
    ax.tick_params(labelsize=20)
    plt.title('Null Brand Emotion Counts', fontsize=30)
    plt.xlabel('', fontsize=20)
    plt.ylabel("")
    # plt.savefig('null_emotion_counts',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def brand_emotions(df):
    emo = df.groupby('brand_product')['emotion']\
            .value_counts().reset_index(name='count')
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='count', y ='brand_product',
                data=emo,  hue='emotion', palette='Blues_r')
    plt.title('Emotion Counts for Brand/Product')
    plt.ylabel('')
    plt.xlabel('')
#     plt.savefig('brand_emotions',  bbox_inches ="tight",\
#     pad_inches = .25, transparent = False)
    plt.show()

def brand_emotion_n(df): 
    emo = df.groupby('brand_product')['emotion']\
            .value_counts(normalize=True).reset_index(name='count')
    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x='count', y ='brand_product', data=emo,
                hue='emotion', palette='Blues_r')
    plt.title('Emotion Percentages for Brand/Product')
    plt.ylabel('')
    plt.xlabel('')
    plt.legend(title='emotion')
#     plt.savefig('brand_emotions_n',  bbox_inches ="tight",\
#     pad_inches = .25, transparent = False)
    plt.show()

def hashtag_c(df):
    counts = df['hashtags'].value_counts()[:20]
    percents = df['hashtags'].value_counts(normalize=True)[:20]
    tags = df['hashtags'].value_counts()[:20].index

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x=counts, y=tags, palette='Blues_r')
    plt.title('Counts of the Top 20 Hashtags')
    plt.xlabel('Count')
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()

def hashtag_p(df):
    counts = df['hashtags'].value_counts()[:20]
    percents = df['hashtags'].value_counts(normalize=True)[:20]
    tags = df['hashtags'].value_counts()[:20].index

    fig, ax = plt.subplots(figsize=(16,8))
    sns.barplot(x=percents, y=tags, palette='Blues_r')
    plt.title('Percentages of the Top 20 Hashtags')
    plt.xlabel('Percent')
    # plt.savefig('brand_emotions',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
    
def top_word_list(data, n):
    "Plots a bargraph of the top words in a corpus."
    
    processed_data = list(map(fn.tokens, data))
    word_li = fn.word_list(processed_data)
    freqdist = FreqDist(word_li)
    most_common = freqdist.most_common(n)
    word_list = [tup[0] for tup in most_common]
    word_counts = [tup[1] for tup in most_common]
    plt.figure(figsize=(15,7))
    sns.barplot(x=word_counts, y=word_list, palette='Blues_r')
    plt.title(f'The Top {n} Words')
    # plt.savefig('title',  bbox_inches ="tight",\
    #             pad_inches = .25, transparent = False)
    plt.show()
    
def word_cloud(data, n):
    "Plots a word cloud of the top n words in a corpus."
    
    processed_data = list(map(fn.tokens, data))
    word_li = fn.word_list(processed_data)
    freqdist = FreqDist(word_li)
    most_common = freqdist.most_common(n)
    word_list = [tup[0] for tup in most_common]
    word_counts = [tup[1] for tup in most_common]
    word_dict = dict(zip(word_list, word_counts))
    plt.figure(figsize=(14,7), facecolor='k')
    wordcloud = WordCloud(colormap='Blues')\
                          .generate_from_frequencies(word_dict)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.tight_layout(pad=0)
#     plt.savefig('aug_pos',  bbox_inches ="tight",\
#                 pad_inches = .25, transparent = False)
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