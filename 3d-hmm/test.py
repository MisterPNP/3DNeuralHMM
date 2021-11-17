import torch

from load_story_cloze import load_cloze_test
from score_cloze_accuracy import score_prediction_batch
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


# TODO: Remove low-frequency tokens?


stories = load_cloze_test()

batch_size = 1 + len(stories) // 5
batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(batch_size)
false_batch = stories[:batch_size, torch.tensor([0, 1, 2, 3, 5])]

# model = Gradient3DHMM(6, 6, 5558)
# learning_rate = 1e-4
# num_epochs = 3

model = Neural3DHMM(6, 6, 5558)
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
