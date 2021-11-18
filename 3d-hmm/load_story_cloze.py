from pprint import pprint
import string
import json
import pickle
import os

import pandas as pd

import nltk.corpus
import nltk.stem
import nltk.tokenize

import torch
import torchtext

def get_processed_data():
    # preprocess_cloze_test('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'story_cloze_2016_test')
    # preprocess_cloze_test('cloze_test_val__spring2016 - cloze_test_ALL_val.csv', 'story_cloze_2016_val')
    preprocess_ROC_test('ROCStories__spring2016 - ROCStories_spring2016.csv', 'ROC_stories_2016')

# HELPER FUNCTIONS

def required_downloads():
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

def get_wordnet_pos(word):
    wordnet = nltk.corpus.wordnet
    # TODO what is this doing?
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize(sentence):
    sentence = sentence.lower()
    stopwords = nltk.corpus.stopwords.words('english')
    tokenizer = nltk.tokenize.WordPunctTokenizer()
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = []
    for token in tokenizer.tokenize(sentence):
        if token[0] in string.punctuation or token in stopwords:
            continue
        pos = get_wordnet_pos(token)
        lemmas.append(lemmatizer.lemmatize(token, pos))
    return lemmas

# gets the data from file and pre-processes it
def preprocess_cloze_test(data_name, output_name):
    word2index = get_word2index()
    filepath = '../data/' + data_name
    df = pd.read_csv(filepath)
    sequences = []  # each set of six sentences
    vocab = {}      # vocab counts across dataset
    for i in range(len(df)):
        sentences = []
        for sentence in df.iloc[i, 1:5]:
            lemmas = list(map(word2index.get, lemmatize(sentence)))
            for lemma in lemmas:
                if lemma not in vocab:
                    vocab[lemma] = 0
                vocab[lemma] += 1
            sentences.append(lemmas)
        # make the correct ending come before the incorrect one
        last_two = df.iloc[i, 5:7] if df.iloc[i, 7] == 1 else [df.iloc[i, 6], df.iloc[i, 5]]
        for sentence in last_two:
            lemmas = list(map(word2index.get, lemmatize(sentence)))
            for lemma in lemmas:
                if lemma not in vocab:
                    vocab[lemma] = 0
                vocab[lemma] += 1
            sentences.append(lemmas)
        sequences.append(sentences)

    with open("../data/" + output_name + "_vocab.voc", "w") as out:
        for lemma, count in vocab.items():
            out.write(f"{lemma}\t{count}\n")

    with open("../data/" + output_name + ".json", "w") as out:
        json.dump(sequences, out, separators=(",", ":"))

    print("done")

# load the data from file if it has already been written by preprocess_cloze_test()
def load_cloze_file(filename):
    # load _correct_ five sentences instead
    with open("../data/"+filename) as file:
        stories = json.load(file)
        sentence_length = 0
        for story in stories:
            for sentence in story:
                sentence_length = max(sentence_length, len(sentence))
        for story in stories:
            for i, sentence in enumerate(story):
                story[i] += [-1] * (sentence_length - len(sentence))
        return torch.tensor(stories)

# gets the data from file and pre-processes it
def preprocess_ROC_test(data_name, output_name):
    word2index = get_word2index()
    filepath = '../data/' + data_name
    df = pd.read_csv(filepath)

    sequences = []  # each set of six sentences
    vocab = {}  # vocab counts across dataset
    for i in range(len(df)):
        sentences = []
        for sentence in df.iloc[i, 2:7]:
            lemmas = list(map(word2index.get, lemmatize(sentence)))
            for lemma in lemmas:
                if lemma not in vocab:
                    vocab[lemma] = 0
                vocab[lemma] += 1
            sentences.append(lemmas)
        sequences.append(sentences)

    with open("../data/" + output_name + "_vocab.voc", "w") as out:
        for lemma, count in vocab.items():
            out.write(f"{lemma}\t{count}\n")
    print("done with vocab")

    with open("../data/" + output_name + ".json", "w", encoding="utf-8") as out:
        json.dump(sequences, out, separators=(",", ":"))
    print("done with json")


def load_roc_test():
    return load_cloze_file("ROC_stories_2016.json")


def load_cloze_test():
    return load_cloze_file("story_cloze_2016_test.json")


def load_cloze_valid():
    return load_cloze_file("story_cloze_2016_val.json")

def make_vocab():
    filename = "../data/glove/glove.6B.100d.txt"
    # TODO if not os.path.exists(filename):
    #     download_glove_embeddings()

    word2index = {}
    with open(filename, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            word2index[line.split()[0]] = i

    with open("../data/word2index.pkl", "wb") as file:
        pickle.dump(word2index, file)

def get_word2index():
    filename = "../data/word2index.pkl"
    if not os.path.exists(filename):
        make_vocab()

    with open(filename, "rb") as file:
        w2i = pickle.load(file)
        w2i[None] = 1  # TODO use an averaged vector for unk
        return w2i
