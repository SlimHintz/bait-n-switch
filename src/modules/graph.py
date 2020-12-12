from nltk.tokenize import RegexpTokenizer
from nltk.probability import FreqDist
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image

def get_vocab_length(Series, show_graph=True):
    
    corpus = " ".join(Series.to_list())
    corpus = tokenizer.tokenize(corpus)
    freqdist = FreqDist(corpus)
    if show_graph:
        freqdist.plot(30)
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