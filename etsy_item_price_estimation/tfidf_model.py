"""
This module is used for transforming text to tfidf vectors.

Code was based on a tutorial:
https://www2.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
"""


import string
import nltk

from sklearn import feature_extraction


def stem_tokens(tokens, stemmer):
    """
    Construct stem using tokens
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text):
    """
    Tokenize the words and return stems
    """
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, nltk.stem.porter.PorterStemmer())
    return stems

def tfidf_transform(text_list, max_features=None):
    """
    Do the actual transforming with tfidf
    """
    token_dict = {}
    # create translator for punctuation removal
    translator = str.maketrans('', '', string.punctuation)
    for i, text_element in enumerate(text_list):
        # convert all words to lower case
        text_lowers = text_element.lower()
        # remove punctuation
        no_punctuation = text_lowers.translate(translator)
        token_dict[str(i)] = no_punctuation
    # this can take some time
    if max_features is None:
        tfidf = feature_extraction.text.TfidfVectorizer(
            tokenizer=tokenize, stop_words='english')
    else:
        tfidf = feature_extraction.text.TfidfVectorizer(
            tokenizer=tokenize, stop_words='english',
            max_features=max_features)

    tfs_vectors = tfidf.fit_transform(token_dict.values())
    return tfs_vectors, tfidf
