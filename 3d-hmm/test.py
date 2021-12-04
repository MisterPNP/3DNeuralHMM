from functools import partial
from itertools import accumulate

import numpy as np

from load_story_cloze import *
from train import train
from score_cloze_accuracy import score_prediction_batch
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


index2word = pickle.load(open("../data/index2word.pkl", 'rb'))

stories = load_roc_test()

batch_size = 1000
batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(batch_size)


# model = Gradient3DHMM(6, 6, len(index2word))
# learning_rate = 1e-2
# num_epochs = 2

model = Neural3DHMM(6, 6, len(index2word), win_size=1)#, token_embeddings=torch.load("../data/word_tensors.tensor"))
learning_rate = 1e-5
num_epochs = 2


analysis = train(model, batches, lr=learning_rate, num_epochs=num_epochs,
                 # negative_batches=stories[:, torch.tensor([0, 1, 2, 3, 5])].split(batch_size),
                 valid_batches=[load_cloze_valid()[:, :5]], accuracy_function=None)

print()
print("DONE LEARNING")

print()
print("TEST_LOSS", analysis['test_loss'])
print("VALID_LOSS", analysis['valid_loss'])
print("ACCURACY", score_prediction_batch(model))


def compose(funcs):
    return list(accumulate(funcs, lambda f, g: lambda x: f(g(x))))[-1]


def pad(n, s):
    return s + " " * (n - len(s))


def print_emissions():
    model.compute_emission_matrix()
    emissions = model.emission_matrix
    highest = emissions.topk(3, dim=-1).indices
    words = map(compose([
        ",".join,
        partial(map, compose([
            index2word.get,
            torch.Tensor.item
        ])),
    ]), highest.flatten(end_dim=-2))
    words = list(words)
    print("EMISSIONS")
    for z in range(model.z_size):
        for y in range(model.xy_size):
            for x in range(model.xy_size):
                index = x + model.xy_size * y + model.xy_size * model.xy_size * z
                print("{:<32}".format(words[index]), end="")
            print()
        print()


print_emissions()
