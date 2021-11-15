import pandas as pd
import numpy as np
import nltk
import re
import contractions
import sklearn

from bs4 import BeautifulSoup as bs


import os
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB


DATASET_DIR='../dataset/'

# try to process part of the poem dataset (all.csv)
def load_data(dataset_dir):

    # read data from all.csv file
    dataset = pd.read_csv(dataset_dir+'all.csv', sep=',', usecols=['content', 'type'])
    # shuffle and reindex the dataset
    dataset = dataset.sample(frac=1.0, random_state=26).reset_index(drop=True)

    labels = np.array(dataset['type'].value_counts().keys(), dtype='str')
    label_map = {}
    for idx in range(len(labels)):
        label_map[labels[idx]] = idx
    #print(labels)
    #print(label_map)

    poem_labels = np.array([label_map[x] for x in dataset['type'].values])
    #print(ptype)

    raw_poems = dataset['content'].values
    #print(pcontent)

    return labels, label_map, poem_labels, raw_poems


def clean_poems(raw_poems):
    def clean_poem(poem):
        # convert to lower case
        poem = poem.lower()
        # perform contractions
        poem = contractions.fix(poem)
        # remove non-alphabetical characters
        poem = re.sub(r'-{2,}', ' ', poem)
        poem = re.sub(r'-{1}', '', poem)
        poem = re.sub(r'[^a-zA-Z]', ' ', poem)
        # remove extra spaces
        poem = re.sub(r'^\s*', '', poem)
        poem = re.sub(r'\s{2,}', ' ', poem)
        # convert to lower case
        poem = poem.lower()
        return poem

    len_before, len_after = 0, 0
    poems_cleaned = []

    for poem in raw_poems:
        len_before += len(poem)
        poem_cleaned = clean_poem(poem)
        len_after += len(poem_cleaned)
        poems_cleaned.append(poem_cleaned)

    #for i in range(3):
    #    print(raw_poems[i])
    #    print('--------')
    #    print(poems_cleaned[i])
    #    print('########################\n')

    poem_num = len(raw_poems)
    print("- average length of peom before and after data cleaning:",
            len_before / poem_num, ",", len_after / poem_num)
    return np.array(poems_cleaned)


def preprocess_poems(poems_cleaned):
    stop_words = set(stopwords.words('english'))
    #stemmer = PorterStemmer()
    # define word tag (noun, verb, adjective, adverb) mapper
    def tag_mapper(tag):
        first_char = tag[1][0]
        if first_char == 'N':
            return wordnet.NOUN
        elif first_char == 'V':
            return wordnet.VERB
        elif first_char == 'J':
            return wordnet.ADJ
        elif first_char == 'R':
            return wordnet.ADV
        else:
            return wordnet.NOUN
    lemmatizer = WordNetLemmatizer()
    
    len_before, len_after = 0, 0
    poems_preprocessed = []

    for poem in poems_cleaned:
        len_before += len(poem)

        tokens = word_tokenize(poem)
        # remove stopwords
        tokens_filtered = [token for token in tokens if not token in stop_words]
        #tokens_stemmed = [stemmer.stem(token) for token in tokens_filtered]
        pos_tags = nltk.pos_tag(tokens_filtered)
        # perform lemmatization
        tokens_lemmatized = [lemmatizer.lemmatize(token, tag_mapper(tag)) \
                            for token, tag in zip(tokens_filtered, pos_tags)]

        poem_preprocessed = ' '.join(tokens_lemmatized)

        len_after += len(poem_preprocessed)
        poems_preprocessed.append(poem_preprocessed)

    poem_num = len(poems_cleaned)
    print("- average length of peom before and after data preprocessing:",
            len_before / poem_num, ",", len_after / poem_num)

    #for i in range(5):
    #    print(poems_preprocessed[i], '\n')

    return np.array(poems_preprocessed)


def extract_tfidf_features(corpus):
    vectorizer = TfidfVectorizer(min_df=10, sublinear_tf=True)
    tfidf_matrix = vectorizer.fit_transform(corpus)
    #feature_names = vectorizer.get_feature_names()
    #print(feature_names)
    print('\ttf-idf vocabulary size: ', len(vectorizer.vocabulary_))
    return tfidf_matrix





if __name__ == '__main__':
    # load raw poems
    labels, label_map, poem_label, raw_poems = load_data(DATASET_DIR)
    
    # preprocess
    poems_cleaned = clean_poems(raw_poems)
    poems_preprocessed = preprocess_poems(poems_cleaned)

    # extract tf-idf features
    matrix = extract_tfidf_features(poems_preprocessed)
