from pprint import pprint
import string

import nltk.corpus
import nltk.stem

def get_wordnet_pos(word):
    wordnet = nltk.corpus.wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def test():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

    sentence = "The striped story-cloze task has recently also been addressed as a shared task at EACL (Mostafazadeh et al., 2017) with a significantly expanded dataset, and achieving much higher performance"
    counts = {}

    stopwords = set(nltk.corpus.stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    for token in nltk.word_tokenize(sentence):
        token = token.lower()
        if token in string.punctuation or token in stopwords:
            continue
        pos = get_wordnet_pos(token)
        token = lemmatizer.lemmatize(token, pos)
        if token not in counts:
            counts[token] = 0
        counts[token] += 1

    pprint(counts)

import pandas as pd

def readClozeTest():
    filepath = '../data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
    df = pd.read_csv(filepath)
