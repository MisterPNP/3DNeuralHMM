import pandas as pd
import torch

from load_story_cloze import load_cloze_test
from train import train
from score_cloze_accuracy import score_prediction_batch
from glove_word_embeddings import get_embeddings
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


# TODO: Remove low-frequency tokens?


stories = load_roc_test()

batch_size = 1 + len(stories) // 5
batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(batch_size)
false_batch = stories[:batch_size, torch.tensor([0, 1, 2, 3, 5])]


# model = Gradient3DHMM(6, 6, 11571)
# learning_rate = 1e-4
# num_epochs = 3


token_embeddings = []
vocab = pd.read_csv("../data/vocab_test.voc", sep="\t", names=('idx', 'word', 'count'))
embeddings = get_embeddings()
for _, row in vocab.iterrows():
    if row['word'] not in embeddings:
        print(row['word'])
        token_embeddings.append(embeddings['the'])
        continue
    emb = embeddings[row['word']]
    token_embeddings.append(emb)
token_embeddings = torch.stack(token_embeddings)

model = Neural3DHMM(6, 6, 11571, token_embeddings=token_embeddings)
learning_rate = 1e-6
num_epochs = 10


analysis = train(model, batches, lr=learning_rate, num_epochs=num_epochs,
      valid_batches=[false_batch], accuracy_function=score_prediction_batch)

print("DONE LEARNING")

print("TEST_LOSS", analysis['test_loss'])
print("VALID_LOSS", analysis['valid_loss'])
print("ACCURACY", analysis['accuracy'])
