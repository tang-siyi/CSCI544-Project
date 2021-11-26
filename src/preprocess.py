import pandas as pd
pd.set_option('max_colwidth',150)
pd.options.mode.chained_assignment = None
import numpy as np
import nltk

DATASET_DIR = '../dataset/'
FINE_DATA_DIR = DATASET_DIR+'fine_data.csv'

raw_data = pd.read_csv(FINE_DATA_DIR)
#print(raw_data.head(3))

data_pt = raw_data[["poem","our_tag"]]
#print(data_pt.head(10))

total_poem = len(data_pt["poem"])
#print("total number poem: ", total_poem)
poem_content = data_pt["poem"].tolist()
#print(poem_content[1])

# convert the poem content into a long string
def reformat(raw_poem):
    poem = raw_poem[1:-1].replace("\"","").split(",")
    sentences = [x.strip(" ").strip("'") for x in poem]
    poem_str = ' '.join(sentences)
    #print(poem_str)
    return poem_str

poem_content = [reformat(x) for x in poem_content]
#print(poem_content[1])
data_pt["poem"] = poem_content
#print(data_pt.head(5))


data_pt_clean = pd.DataFrame(columns=['poem', 'label'])
# convert label from string to integer
labels = data_pt['our_tag'].value_counts().keys()
label_map = {}
for i in range(len(labels)):
    label_map[labels[i]] = i
data_pt_clean['label'] = [label_map[x] for x in data_pt['our_tag']]



# clean poem 
import re
import contractions
from bs4 import BeautifulSoup as bs
def clean(poem):
    # convert to lower case
    poem = poem.lower()
    # remove HTML and URLs
    poem = bs(poem, features="html.parser").get_text(separator=' ')
    # perform contractions
    poem = contractions.fix(poem)
    # removew non-alphabetical characters
    poem = re.sub(r'-{2,}', ' ', poem)
    poem = re.sub(r'-{1}', '', poem)
    poem = re.sub(r'[^a-zA-Z]', ' ', poem)
    # remove extra spaces
    poem = re.sub(r'^\s*', '', poem)
    poem = re.sub(r'\s{2,}', ' ', poem)
    # convert to lower case
    poem = poem.lower()
    return poem

data_pt_clean['poem'] = [clean(x) for x in data_pt['poem']]
print(data_pt_clean.head(5))


#remove stop word
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
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
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(poem):
    tokens = word_tokenize(poem)
    # remove stopwords
    tokens_filtered = [token for token in tokens if not token in stop_words]
    #tokens_stemmed = [stemmer.stem(token) for token in tokens_filtered]
    pos_tags = nltk.pos_tag(tokens_filtered)
    # perform lemmatization
    tokens_lemmatized = [lemmatizer.lemmatize(token, tag_mapper(tag)) \
                        for token, tag in zip(tokens_filtered, pos_tags)]
    #print(len(tokens_lemmatized))
    poem_preprocessed = ' '.join(tokens_lemmatized)
    return poem_preprocessed

data_pt_clean['poem'] = [preprocess(x) for x in data_pt_clean['poem']]

print("sample poem after data clean and preprogress")
print(data_pt_clean.head(3))


#data_pt_clean.to_csv(DATASET_DIR+'fine_data_clean.csv', index=False)
