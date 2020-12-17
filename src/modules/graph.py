from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import numpy as np
from nltk.corpus import stopwords
import os
import sys
from textblob import TextBlob

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

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

# Import module
module_path = os.path.abspath(os.path.join('./src'))
if module_path not in sys.path:
    sys.path.append(module_path)
    
from modules import preprocessing as pp
from modules import graph, modelling 



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

def show_class_imbalance(df, title='Class Imbalance', PATH=None):
    plt.bar(x=['Normal', 'Clickbait'], height=df.groupby(['target']).target.count());
    plt.title(title)
    plt.ylabel("Document Count", size = 14)
    plt.xlabel("Class", size = 14)
    if PATH:
        plt.savefig(PATH, bbox_inches="tight")
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

# ================================= Viz stopword differences  ================================= 


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

def proportion_with_cardinals(df, PATH):
    
    df_test = df.copy()
    df_test['cardinal'] = df.title.apply(contains_cardinal)

    click = df_test[df_test.target == 1]
    non = df_test[df_test.target == 0]
    click = click.groupby(['cardinal']).target.count()
    non = non.groupby(['cardinal']).target.count()
    
    non = non[1]/non[0] * 100
    click = click[1]/click[0] * 100
    # plot the results
    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=['Normal', "Clickbait"], y=[non, click], ax=ax)
    plt.title("Percent of Titles Containing Cardinal Numbers", size = 16)
    plt.xlabel("Classes", size=15)
    plt.ylabel("Percent %", size = 15)
    if PATH:
        plt.savefig(PATH, bbox_inches="tight")
    
    return ax



# ================================= Viz title false positives/negatives ================================= 

def get_false_positives(predictions, y_test):
    """
    Returns a numpy array of index matched false negatives
    predictions --> binary or bool
    y_test --> binary or bool
    theshold 
    
    returns a np.array
    """
    comparisons = list(zip(y_test, predictions))
    return np.array([1 if (true == 0 and prediction == 1) else 0 for true, prediction in comparisons])

def get_false_negatives(predictions, y_test):
    """
    Returns a numpy array of index matched false negatives
    predictions --> binary or bool
    y_test --> binary or bool
    theshold 
    
    returns a np.array
    """
    comparisons = list(zip(y_test, predictions))
    return np.array([1 if (true == 1 and prediction == 0) else 0 for true, prediction in comparisons])




# ================================= Viz word clouds ================================= 

def generate_wordcloud(dict_, title='WordCloud', PATH=None):
    wordcloud  = WordCloud(min_font_size=10).generate_from_frequencies(dict_)
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.title(title)
    plt.tight_layout(pad = 0) 
    if PATH:
        plt.savefig(PATH, bbox_inches="tight")
    plt.show() 
    

    
    
# ================================= Viz Difference and Intersection Word Clouds ================================= 

def get_intersect(df):
    click_corpus = " ".join(df[df.target==1].title.to_list())
    non_corpus = " ".join(df[df.target==0].title.to_list())
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    
    click_tokenized = tokenizer.tokenize(click_corpus)
    non_tokenized = tokenizer.tokenize(non_corpus)

    fil_click = [word.lower() for word in click_tokenized if word.lower() not in pp.stop_words]
    fil_non = [word.lower() for word in non_tokenized if word.lower() not in pp.stop_words]
    
    non_set = set(fil_non)
    click_set = set(fil_click)
    
    return click_set.intersection(non_set), click_tokenized

def get_difference(df):
    
    """
    Given a dataframe, returns a set of words that are unique to the the clickbait data set and then a list of all the words 
    used in the clickbait dataset
    """
    click_corpus = " ".join(df[df.target==1].title.to_list())
    non_corpus = " ".join(df[df.target==0].title.to_list())
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    
    click_tokenized = tokenizer.tokenize(click_corpus)
    non_tokenized = tokenizer.tokenize(non_corpus)

    fil_click = [word.lower() for word in click_tokenized if word.lower() not in pp.stop_words]
    fil_non = [word.lower() for word in non_tokenized if word.lower() not in pp.stop_words]
    
    non_set = set(fil_non)
    click_set = set(fil_click)
    
    return click_set.difference(non_set), click_tokenized # This should return nothing

def countX(lst, x): 
    return lst.count(x) 
     
def visualize_intersection(df):
    
    click_corpus = " ".join(df[df.target==1].title.to_list())
    non_corpus = " ".join(df[df.target==0].title.to_list())
    
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    
    click_tokenized = tokenizer.tokenize(click_corpus)
    non_tokenized = tokenizer.tokenize(non_corpus)

    fil_click = [word.lower() for word in click_tokenized if word.lower() not in pp.stop_words]
    fil_non = [word.lower() for word in non_tokenized if word.lower() not in pp.stop_words]
    
    non_set = set(fil_non)
    click_set = set(fil_click)
    
    # Generate sets of words
    diff = non_set.difference(click_set)
    overlap = non_set.intersection(click_set)
    
    # Generate word clouds of each
    def countX(lst, x): 
        return lst.count(x) 
    def freq_of_specific_words(tokenized_list, list_of_interest):
        freqDict = {}
        for word in list_of_interest:
            freqDict[word] = countX(tokenized_list, word)


    non_diff = {}
    for word in diff:
        non_diff[word] = countX(list(non_set), word)
    difference_frequency = sorted(non_diff.items(), reverse=True, key = (lambda x: x[1]))
    
    wordcloud  = WordCloud(min_font_size=10).generate_from_frequencies(dict(difference_frequency))
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.title("Difference Between Click-Bait and Normal Headlines")
    plt.tight_layout(pad = 0) 

    plt.show() 
    
    
    click_diff = {}
    for word in overlap:
        click_diff[word] = countX(fil_click, word)

    difference_frequency = sorted(click_diff.items(), reverse=True, key = (lambda x: x[1]))
    wordcloud  = WordCloud(min_font_size=5).generate_from_frequencies(dict(difference_frequency))
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.title("Intersection Between Click-Bait and Normal Headlines")
    plt.tight_layout(pad = 0) 

    plt.show() 
    
    
    
    
    
# ================================= Viz Evaluation Metrics =================================     
    
    
def plot_cmatrix(actual, predictions, model, PATH = None):
    '''Takes in arrays of actual binary values and model predictions and generates and plots a confusion matrix'''
    cmatrix = confusion_matrix(actual, predictions)

    fig, ax = plt.subplots(figsize = (12,6))
    sns.heatmap(cmatrix, annot=True, fmt='g', ax=ax, cmap='Blues')
    ax.set_xticklabels(['Normal', 'Clickbait'])
    ax.set_yticklabels(['Normal', 'Clickbait'])
    ax.set_ylabel('Actual', size=15)
    ax.set_xlabel('Predicted', size=15)
    ax.set_title(f'Confusion Matrix for {model} Predictions', size =18)
    if PATH:
        plt.savefig(PATH, bbox_inches = "tight")
  
    return plt.show()

def plot_roc_curve(actual, predictions, model = "ROC Curve", PATH=None):
    '''Takes in arrays of actual binary values and model predictions and generates and plots an ROC curve'''
    
    fpr, tpr, threshholds = roc_curve(actual, predictions)
    
    sns.set_style('darkgrid', {'axes.facecolor': '0.9'})
    
    print('AUC: {}'.format(auc(fpr, tpr)))
    plt.figure(figsize=(10, 8))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label=model)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    try:
        plt.ylim([0.0, 1.05])
    except:
        print("plt.ylim throwing an type error")
    plt.yticks([i/20.0 for i in range(21)])
    plt.xticks([i/20.0 for i in range(21)])
    plt.xlabel('False Positive Rate', size=15)
    plt.ylabel('True Positive Rate', size=15)
    plt.title(f'{model} ROC Curve')
    plt.legend(loc='lower right')
    if PATH:
        plt.savefig(PATH, bbox_inches='tight')
    return plt.show()