from pprint import pprint
import string
import json
import pickle
import os
import urllib
import zipfile

import pandas as pd

import nltk.corpus
import nltk.stem
import nltk.tokenize

import torch


def get_processed_data():
    required_downloads()
    make_vocab()
    preprocess_stories(
        'cloze_test_test__spring2016 - cloze_test_ALL_test.csv',
        'story_cloze_2016_test', make_cloze_sentences)
    preprocess_stories(
        'cloze_test_val__spring2016 - cloze_test_ALL_val.csv',
        'story_cloze_2016_val', make_cloze_sentences)
    preprocess_stories(
        'ROCStories__spring2016 - ROCStories_spring2016.csv',
        'ROC_stories_2016', make_roc_sentences)


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
def preprocess_stories(data_name, output_name, make_sentences):
    filepath = '../data/' + data_name
    df = pd.read_csv(filepath)

    word2index = get_word2index()
    def lemma2index(lemma):
        return word2index[lemma] if lemma in word2index else word2index["<unk>"]

    sequences = []  # each set of six sentences
    vocab = {}      # vocab counts across dataset
    for i in range(len(df)):
        sequences.append(make_sentences(df.iloc[i], vocab, lemma2index))

    with open("../data/" + output_name + "_vocab.voc", "w") as out:
        for lemma, count in vocab.items():
            out.write(f"{lemma}\t{count}\n")
    print("done with vocab")

    with open("../data/" + output_name + ".json", "w", encoding="utf-8") as out:
        json.dump(sequences, out, separators=(",", ":"))
    print("done with json")


def make_cloze_sentences(row, vocab, lemma2index):
    # make the correct ending come before the incorrect one
    if row[7] == 2:
        row[5], row[6] = row[6], row[5]
    return make_sentences_from_row(row[1:7], vocab, lemma2index)


def make_roc_sentences(row, vocab, lemma2index):
    return make_sentences_from_row(row[2:7], vocab, lemma2index)


def make_sentences_from_row(row, vocab, lemma2index):
    sentences = []
    for sentence in row:
        lemmas = list(map(lemma2index, lemmatize(sentence)))
        for lemma in lemmas:
            if lemma not in vocab:
                vocab[lemma] = 0
            vocab[lemma] += 1
        sentences.append(lemmas)
    return sentences


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


def load_roc_test():
    return load_cloze_file("ROC_stories_2016.json")


def load_cloze_test():
    return load_cloze_file("story_cloze_2016_test.json")


def load_cloze_valid():
    return load_cloze_file("story_cloze_2016_val.json")


def download_glove_embeddings():
    filename = '../data/glove.6B.zip'
    if not os.path.exists(filename):
        print("downloading GLOVE embeddings")
        urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename)

    print("extracting GLOVE embeddings")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        # for fileinfo in zip_ref.infolist():
        #     filename = fileinfo.filename
        #     outputfile = open(filename, "wb")
        #     with zip_ref.open(filename) as inputfile:
        #         outputfile.write(inputfile.read())
        zip_ref.extractall('../data/glove')


def make_vocab():
    glove2index = {}
    with open("../data/vocab_frequent.voc", "r") as file:
        for i, line in enumerate(file):
            glove_index = int(line.split()[0])
            glove2index[glove_index] = i + 1

    word2index = {}
    index2word = {}
    index2tensor = {}
    glove = "../data/glove/glove.6B.100d.txt"
    if not os.path.exists(glove):
        download_glove_embeddings()
    with open(glove, "r", encoding="utf-8") as file:
        for i, line in enumerate(file):
            if i in glove2index:
                word = line.split()[0]
                index = glove2index[i]
                tensor = torch.tensor(list(map(float, line.split()[1:])))
                word2index[word] = index
                index2word[index] = word
                index2tensor[index] = tensor

    tensors = torch.stack(list(map(
        lambda i: index2tensor[i + 1], range(len(index2tensor))
    )))
    # use average of tensors as <unk> embedding
    tensors = torch.cat([tensors.mean(0).unsqueeze(0), tensors])
    word2index["<unk>"] = 0
    index2word[0] = "<unk>"

    with open("../data/word2index.pkl", "wb") as file:
        pickle.dump(word2index, file)
    with open("../data/index2word.pkl", "wb") as file:
        pickle.dump(index2word, file)
    torch.save(tensors, "../data/word_tensors.tensor")


def get_word2index():
    filename = "../data/word2index.pkl"
    if not os.path.exists(filename):
        make_vocab()

    with open(filename, "rb") as file:
        return pickle.load(file)
