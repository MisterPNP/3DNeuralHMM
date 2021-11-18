import os
import zipfile
import torch
import urllib.request


def download_glove_embeddings():
    print("downloading GLOVE embeddings")
    urllib.request.urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", "../data/glove.6B.zip")

    print("extracting GLOVE embeddings")
    with zipfile.ZipFile('../data/glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall('../data/glove')

def get_embeddings():
    filename = '../data/glove/glove.6B.100d.txt'

    if not os.path.exists(filename):
        download_glove_embeddings()

    print("loading GLOVE embeddings from file")
    embeddings = {}
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddings[word] = torch.tensor(list(map(float, values[1:])))
    return embeddings
