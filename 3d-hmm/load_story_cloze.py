from pprint import pprint
import string
import json

import pandas as pd

import nltk.corpus
import nltk.stem
import nltk.tokenize

import torch
import torchtext

def get_processed_data():
    # TODO need this?
    return load_cloze_test()

    #preprocess_cloze_test('cloze_test_test__spring2016 - cloze_test_ALL_test.csv', 'story_cloze_2016_test')
    #preprocess_cloze_test('cloze_test_val__spring2016 - cloze_test_ALL_val.csv', 'story_cloze_2016_val')

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
    filepath = '../data/' + data_name
    df = pd.read_csv(filepath)

    # TODO use validation file in constructing vocabulary?
    #      (kind of depends on if we're going to use pretrained word embeddings...)
    sequences = []  # each set of six sentences
    vocab = {}      # vocab counts across dataset
    for i in range(len(df)):
        sentences = []
        for sentence in df.iloc[i, 1:5]:
            lemmas = lemmatize(sentence)
            for lemma in lemmas:
                if lemma not in vocab:
                    vocab[lemma] = 0
                vocab[lemma] += 1
            sentences.append(lemmas)
        # make the correct ending come before the incorrect one
        last_two = df.iloc[i, 5:7] if df.iloc[i, 7] == 1 else [df.iloc[i, 6], df.iloc[i, 5]]
        for sentence in last_two:
            lemmas = lemmatize(sentence)
            for lemma in lemmas:
                if lemma not in vocab:
                    vocab[lemma] = 0
                vocab[lemma] += 1
            sentences.append(lemmas)
        sequences.append(sentences)

    # map from lemmas to indexes
    lemma_to_i = {}
    for i, lemma in enumerate(vocab):
        lemma_to_i[lemma] = i
    for sentences in sequences:
        for lemmas in sentences:
            for i, lemma in enumerate(lemmas):
                lemmas[i] = lemma_to_i[lemma]

    with open("../data/" + output_name + "_vocab.voc", "w") as out:
        for i, (lemma, count) in enumerate(vocab.items()):
            out.write("{}\t{}\t{}\n".format(i, lemma, count))

    with open("../data/" + output_name + ".json", "w") as out:
        json.dump(sequences, out, separators=(",", ":"))

    print("done")

# load the data from file if it has already been written by preprocess_cloze_test()
def load_cloze_test():
    # load _correct_ five sentences instead
    with open("../data/test.json") as file:
        stories = json.load(file)
        sentence_length = 0
        for story in stories:
            for sentence in story:
                sentence_length = max(sentence_length, len(sentence))
        for story in stories:
            for i, sentence in enumerate(story):
                story[i] += [-1] * (sentence_length - len(sentence))
        return torch.tensor(stories)












