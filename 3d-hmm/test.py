import torch

from load_story_cloze import load_cloze_test
from scalar_3d_hmm_model import Scalar3DHMM

stories = load_cloze_test()
print(stories.shape)

batches = stories[:, torch.tensor([0,1,2,3,4])].split(64)
print(len(batches), batches[0].shape)

false_batch = stories[:64, torch.tensor([0,1,2,3,5])]

model = Scalar3DHMM(6, 6, 5558)

# SGD
lr = 0.001
for epoch in range(1):
    for idx, batch in enumerate(batches):
        print(idx)
        p = model.score(batch, 5, 0)
        p.sum(-1).backward()
        with torch.no_grad():
            print(p.mean())
            print(model.score(false_batch, 5, 0).mean())

            model.emission_matrix_unnormalized += lr * model.emission_matrix_unnormalized.grad
            model.emission_matrix_unnormalized.grad.zero_()
            model.transition_matrix_unnormalized += lr * model.transition_matrix_unnormalized.grad
            model.transition_matrix_unnormalized.grad.zero_()
            model.state_priors_unnormalized += lr * model.state_priors_unnormalized.grad
            model.state_priors_unnormalized.grad.zero_()
print("DONE")

# TODO backward algorithm, instead of SGD
# ...
