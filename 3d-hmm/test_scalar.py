import pickle

import numpy as np
import torch

from load_story_cloze import load_cloze_test, load_roc_test
from score_cloze_accuracy import score_prediction_batch
from scalar_3d_hmm_model import Scalar3DHMM
from scalar_hmm_model import ScalarHMM


# TODO backward algorithm, instead of SGD

index2word = pickle.load(open("../data/index2word.pkl", 'rb'))

# model = Scalar3DHMM(4, 6, len(index2word))
model = ScalarHMM(10, len(index2word))
# stories_correct = load_cloze_test()[:, torch.tensor([0, 1, 2, 3, 4])]
stories_correct = load_roc_test()[:100]

with torch.no_grad():
    def pad(s, n):
        return s + " " * (n - len(s))

    def print_emissions():
        highest = model.emissions.topk(3, dim=-1).indices.numpy()
        words = np.apply_along_axis(lambda r: pad(",".join(map(index2word.get, r)), 64), -1, highest)
        print("EMISSIONS")
        for i, word in enumerate(words):
            print(f"<state {i}> {word}")

    print(model.score(stories_correct, 5).mean())
    print_emissions()
    for epoch in range(10):
        priors, transitions, emissions = model.baum_welch_updates(stories_correct)
        model.priors = priors
        model.transitions = transitions
        model.emissions = emissions
        print(model.score(stories_correct, 5).mean())
        print_emissions()

    print(score_prediction_batch(model))
