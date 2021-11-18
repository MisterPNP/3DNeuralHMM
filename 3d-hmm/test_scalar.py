import torch

from load_story_cloze import load_cloze_test
from score_cloze_accuracy import score_prediction_batch
from scalar_3d_hmm_model import Scalar3DHMM


# TODO backward algorithm, instead of SGD

model = Scalar3DHMM(2, 1, 5558)
stories = load_cloze_test()
stories_correct = stories[:, torch.tensor([0, 1, 2, 3, 4])]

with torch.no_grad():
    print(model.score(stories_correct, 5).mean())
    for epoch in range(3):
        priors, transitions, emissions = model.baum_welch_updates(stories_correct)
        model.priors = priors
        model.transitions = transitions
        model.emissions = emissions
        print(model.score(stories_correct, 5).mean())

    print(score_prediction_batch(model, first=0, last=1000))
