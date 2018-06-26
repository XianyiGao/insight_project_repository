import nltk
import string
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, PorterStemmer())
    return stems

def tfidf_transform(textList, max_features=None):
    token_dict = {}
    # create translator for punctuation removal
    translator = str.maketrans('', '', string.punctuation)
    for i in range(0, len(textList)):
        text = textList[i]
        # convert all words to lower case
        lowers = text.lower()
        # remove punctuation
        no_punctuation = lowers.translate(translator)
        token_dict[str(i)] = no_punctuation

    # this can take some time
    if max_features == None:
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    else:
        tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=max_features)
    
    tfs = tfidf.fit_transform(token_dict.values())
    return tfs, tfidf

