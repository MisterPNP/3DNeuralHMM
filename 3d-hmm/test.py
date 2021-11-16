import torch

from load_story_cloze import load_cloze_test
from scalar_3d_hmm_model import Scalar3DHMM

stories = load_cloze_test()
print(stories.shape)

batches = stories[:, torch.tensor([0,1,2,3,4])].split(64)
print(len(batches), batches[0].shape)

false_batch = stories[:64, torch.tensor([0,1,2,3,5])]

model = Scalar3DHMM(7, 6, 5558)
for epoch in range(50):
    for idx, batch in enumerate(batches):
        print(idx)
        p = model.forward(batch, 5, 0)
        with torch.no_grad():
            print(p.mean())
            print(model.forward(false_batch, 5, 0).mean())
        p.sum(-1).backward()  # TODO is this correct?
print("DONE")
