from pprint import pprint
import string
import json

import pandas as pd

import nltk.corpus
import nltk.stem
import nltk.tokenize

import torchtext

def get_wordnet_pos(word):
    wordnet = nltk.corpus.wordnet
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def test():
    return loadClozeTest()

    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

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

def preprocessClozeTest():
    filepath = '../data/cloze_test_test__spring2016 - cloze_test_ALL_test.csv'
    df = pd.read_csv(filepath)

    # TODO use validation file in constructing vocabulary?
    #      (kind of depends on if we're going to use pretrained word embeddings...)
    sequences = []  # each set of six sentences
    vocab = {}      # vocab counts across dataset
    for i in range(len(df)):
        sentences = []
        for sentence in df.iloc[i, 1:7]:
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

    with open("../data/vocab_test.voc", "w") as out:
        for i, (lemma, count) in enumerate(vocab.items()):
            out.write("{}\t{}\t{}\n".format(i, lemma, count))

    with open("../data/test.json", "w") as out:
        json.dump(sequences, out, separators=(",", ":"))

    print("done")

def loadClozeTest():
    with open("../data/test.json") as file:
        sequences = json.load(file)
        examples = torchtext.data.Example.fromlist(
            sequences,
            []
        )
        return examples

    # want to end up with (batches of) torch tensor
    # with [
    #        [
    #           [i i i i i]
    #           [i i i _ _]
    #           [i i i i _]
    #           [i i i i i]
    #           [i i i _ _]
    #           [i i i i _]
    #        ]
    #        [
    #           [i i i i i]
    #           [i i i _ _]
    #           [i i i i _]
    #           [i i i i i]
    #           [i i i _ _]
    #           [i i i i _]
    #        ]
    #        ...
    #     ]
