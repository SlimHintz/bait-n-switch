from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import numpy as np
from nltk.corpus import stopwords

from textblob import TextBlob

stop_words = set(stopwords.words("english"))
# Extend stopwords (see analysis below)
extension = {
    'trumps',
    'trump',
    'obama',
    'donald',
    'new',
    'u'
}
stop_words.update(extension)

tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

def get_vocab_length(Series, words=30, title="Word Frequency", show_graph=True):
    corpus = " ".join(Series.to_list())
    corpus = tokenizer.tokenize(corpus)
    freqdist = FreqDist(corpus)
    if show_graph:
        fig, ax = plt.subplots(nrows=1,ncols=1, figsize=(12,6))
        freqdist.plot(words, title= title)
    print(f"Current Vocab size is = {len(freqdist)}")
    return freqdist


def get_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment[1]

def get_polarity(text):
    blob = TextBlob(text)
    return blob.sentiment[0]


def countX(lst, x): 
    return lst.count(x) 

def freq_of_specific_words(tokenized_list, list_of_interest):
    freqDict = {}
    for word in list_of_interest:
        freqDict[word] = countX(tokenized_list, word)
    return freqDict

def show_wordcloud(dictionary, title, min_font = 10):
    wordcloud  = WordCloud(min_font_size=min_font).generate_from_frequencies(dictionary)
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off")
    if title:
        plt.title(title)
    else:
        plt.title("Word Cloud")
    plt.tight_layout(pad = 0) 

    plt.show() 

def show_class_imbalance(df, title='Class Imbalance'):
    plt.bar(x=['Normal', 'Clickbait'], height=df.groupby(['target']).target.count(), color='b');
    plt.title(title)
    plt.ylabel("Document Count")
    plt.show()

# ================================= Viz Average word length  =================================    
def get_average_word_length(title):
    return np.mean([len(word) for word in title.split()])


def word_lengths(df, ax=None, content='title', title='data', x_lim = [0,10]):
    click_len = df[df.target == 1][content].apply(get_average_word_length)
    non_len = df[df.target == 0][content].apply(get_average_word_length)
    
    if not ax:
        fig, ax = plt.subplots()
        for a, b in zip([click_len, non_len], ['Clickbait', 'Non-clickbait']):
            sns.distplot(a, bins=50, ax=ax, kde=True,  label=b)
            ax.legend()
        ax.set_xlim(x_lim)
        ax.set_xlabel("Average Word Length", size = 14)
        ax.set_title(f"Distribution of Title Length Between \n Clickbait and Non-Clickbait News Headlines {title}", size =17)
        return ax;
    else:
        for a, b in zip([click_len, non_len], ['Clickbait', 'Non-clickbait']):
            sns.distplot(a, bins=50, ax=ax, kde=False,  label=b)
            ax.legend()
        ax.set_xlim(x_lim)
        ax.set_xlabel("Average Word Length", size=14)
        ax.set_title(f"Distribution of Title Length Between \n Clickbait and Non-Clickbait News Headlines {title}", size =17)
        return ax;


# ================================= Viz title Length  =================================    
def get_len(string):
    return len(tokenizer.tokenize(string))

def title_lengths(df, ax, content='title', title='data', x_lim = [0,100]):
    click_len = df[df.target == 1][content].apply(get_len)
    non_len = df[df.target == 0][content].apply(get_len)

    for a, b in zip([click_len, non_len], ['Clickbait', 'Non-clickbait']):
        sns.distplot(a, bins=50, ax=ax, kde=False,  label=b)
        ax.legend()
    ax.set_xlim(x_lim)
    ax.set_xlabel("Length of Title (words)")
    ax.set_title(f"Distribution of Title Length Between \n Clickbait and Non-Clickbait News Headlines {title}", size =10)
    return ax;

# ================================= Viz title Length  ================================= 


def remove_stopwords_tokenized(title):
    return ([word.lower() for word in tokenizer.tokenize(title) if word.lower() not in stop_words])

def stopword_proportion(title):
    tokenized = tokenizer.tokenize(title)
    return (len(tokenized) + 1)/(len(remove_stopwords_tokenized(title)) + 1)

def stopword_hist(df, stop_words, ax):
        click_props = df[df.target == 1].title.apply(stopword_proportion)
        non_props = df[df.target == 0].title.apply(stopword_proportion)
        for a, b in zip([non_props, click_props], ['Normal','Clickbait']):
            sns.distplot(a, bins=30, ax=ax, kde=True,  label=b)
            ax.legend()
        ax.set_xlim([0.5,3])
        ax.set_xlabel("Proportion of Stopwords")
        ax.set_title(f"Proportion of Stopwords Between \n Clickbait and Non-Clickbait News Headlines", size =15)
        return ax;


def stopword_bar(df, stop_words, ax):
    df_test = df.copy()
    df_test['prop'] = df.title.apply(stopword_proportion)
    sns.barplot(data=df_test, x='target', y='prop', ax=ax, ci=False)
    ax.set_title("Proportion of Stopwords between classes")
    ax.set_ylim([1,2])
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Article Class")
    plt.xticks(ticks=range(2),labels=['Normal','Clickbait'])
    return ax

# ================================= Viz title Cardinality  ================================= 

def contains_cardinal(title):
    return any(char.isdigit() for char in title)

def proportion_with_cardinals(df):
    
    df_test = df.copy()
    df_test['cardinal'] = df.title.apply(contains_cardinal)

    click = df_test[df_test.target == 1]
    non = df_test[df_test.target == 0]
    click = click.groupby(['cardinal']).target.count()
    non = non.groupby(['cardinal']).target.count()
    
    non = non[1]/non[0] * 100
    click = click[1]/click[0] * 100
    # plot the results
    ax = sns.barplot(x=['Normal', "Clickbait"], y=[non, click])
    plt.title("Percent of Titles Containing Cardinal Numbers", size = 16)
    plt.xlabel("Classes", size=15)
    plt.ylabel("Percent %", size = 15)
    
    return ax