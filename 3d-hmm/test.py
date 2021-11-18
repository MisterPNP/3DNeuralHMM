import pandas as pd
import torch

from load_story_cloze import load_cloze_test
from score_cloze_accuracy import score_prediction_batch
from glove_word_embeddings import get_embeddings
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


# TODO: Remove low-frequency tokens?


stories = load_cloze_test()

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


# SGD
for epoch in range(num_epochs):
    for idx, batch in enumerate(batches):
        print(idx)
        p = model.score(batch, 5)  # - model.emission_log_p(batch[:, -1]).logsumexp(-1)
        p.sum(-1).backward()

        with torch.no_grad():
            print(p.mean())
            p_false = model.score(false_batch, 5)  # - model.emission_log_p(false_batch[:, -1]).logsumexp(-1)
            print(p_false.mean())

            for parameter in model.parameters():
                # print(parameter.grad.norm())
                parameter += learning_rate * parameter.grad
                parameter.grad.zero_()

print("DONE LEARNING")

print(score_prediction_batch(model, 0, 1000))
