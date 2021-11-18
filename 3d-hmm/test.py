import pandas as pd
import torch

from load_story_cloze import *
from train import train
from score_cloze_accuracy import score_prediction_batch
from glove_word_embeddings import get_embeddings
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


# TODO: Remove low-frequency tokens?


stories = load_roc_test()

batch_size = 1000
batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(batch_size)
# false_batch = stories[:, torch.tensor([0, 1, 2, 3, 5])]

# model = Gradient3DHMM(6, 6, 19477)
# learning_rate = 1e-4
# num_epochs = 3


# token_embeddings = []
# vocab = pd.read_csv("../data/vocab_test.voc", sep="\t", names=('idx', 'count'))
vocab = pd.read_csv("../data/ROC_stories_2016_vocab.voc", sep="\t", names=('idx', 'count'))

token_embeddings = torch.stack(get_embeddings())

model = Neural3DHMM(6, 6, len(token_embeddings), token_embeddings=token_embeddings)
learning_rate = 1e-6
num_epochs = 1


analysis = train(model, batches, lr=learning_rate, num_epochs=num_epochs,
      valid_batches=[], accuracy_function=None)

print()
print("DONE LEARNING")
print()
print("TEST_LOSS", analysis['test_loss'])
print("VALID_LOSS", analysis['valid_loss'])
print("ACCURACY", score_prediction_batch(model))
