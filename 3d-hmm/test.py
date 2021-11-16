import torch

from load_story_cloze import load_cloze_test
from scalar_3d_hmm_model import Scalar3DHMM

stories = load_cloze_test()
print(stories.shape)

batches = stories[:20, :5].split(4)
print(len(batches), batches[0].shape)

model = Scalar3DHMM(15, 6, 5557)
for batch in batches[:1]:
    with torch.no_grad():
        model.forward(batch, 5, 0)
