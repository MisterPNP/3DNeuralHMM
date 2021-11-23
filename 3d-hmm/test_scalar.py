import pickle

import torch

from load_story_cloze import load_cloze_test, load_roc_test
from score_cloze_accuracy import score_prediction_batch
from scalar_3d_hmm_model import Scalar3DHMM
from scalar_hmm_model import ScalarHMM


# TODO backward algorithm, instead of SGD

index2word = pickle.load(open("../data/index2word.pkl", 'rb'))

# model = Scalar3DHMM(4, 6, len(index2word))
model = ScalarHMM(20, len(index2word))
# stories_correct = load_cloze_test()[:, torch.tensor([0, 1, 2, 3, 4])]
stories_correct = load_roc_test()

with torch.no_grad():
    print(model.score(stories_correct, 5).mean())
    for epoch in range(1):
        priors, transitions, emissions = model.baum_welch_updates(stories_correct)
        model.priors = priors
        model.transitions = transitions
        model.emissions = emissions
        print(model.score(stories_correct, 5).mean())

    print(score_prediction_batch(model))
