

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