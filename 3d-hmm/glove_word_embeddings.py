import os
import zipfile
import torch
import urllib.request


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


def get_embeddings():
    filename = '../data/glove/glove.6B.100d.txt'
    if not os.path.exists(filename):
        download_glove_embeddings()

    print("loading GLOVE embeddings from file")
    with open(filename, "r", encoding="utf-8") as f:
        def line2tensor(line):
            return torch.tensor(list(map(float, line.split()[1:])))
        return list(map(line2tensor, f))
    #     for line in f:
    #         values = line.split()
    #         word = values[0]
    #         embeddings[word] = torch.tensor(list(map(float, values[1:])))
    # return embeddings
