
# coding: utf-8

# In[1]:


import nltk
import string
import os

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

def tfidf_transform(textList):
    token_dict = {}
    
    translator = str.maketrans('', '', string.punctuation)
    for i in range(0, len(textList)):
        text = textList[i]
        lowers = text.lower()
        str.maketrans('', '', string.punctuation)
        no_punctuation = lowers.translate(translator)
        token_dict[str(i)] = no_punctuation

    #this can take some time
    tfidf = TfidfVectorizer(tokenizer=tokenize, stop_words='english', max_features=100)
    tfs = tfidf.fit_transform(token_dict.values())
    return tfs

