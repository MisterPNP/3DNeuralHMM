from functools import partial
from itertools import accumulate
import pickle

import numpy as np
import torch

from load_story_cloze import load_cloze_test, load_roc_test, get_processed_data
from score_cloze_accuracy import score_prediction_batch
from scalar_3d_hmm_model import Scalar3DHMM
from scalar_hmm_model import ScalarHMM


# backward algorithm, instead of SGD

# get_processed_data()
index2word = pickle.load(open("../data/index2word.pkl", 'rb'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
model = Scalar3DHMM(6, 6, len(index2word), win_size=1).to(device)
# model = ScalarHMM(15, len(index2word))
# stories_correct = load_cloze_test()[:, torch.tensor([0, 1, 2, 3, 4])]
stories_correct = load_roc_test()[:].to(device)

with torch.no_grad():
    def compose(funcs):
        return list(accumulate(funcs, lambda f, g: lambda x: f(g(x))))[-1]

    def pad(n, s):
        return s + " " * (n - len(s))

    def print_emissions():
        emissions = model.emission_matrix_precomputed
        highest = emissions.topk(3, dim=-1).indices
        words = map(compose([
            ",".join,
            partial(map, compose([
                index2word.get,
                torch.Tensor.item
            ])),
        ]), highest.flatten(end_dim=-2))
        words = list(words)
        print("EMISSIONS")
        for z in range(model.z_size):
            for y in range(model.xy_size):
                for x in range(model.xy_size):
                    index = x + model.xy_size * y + model.xy_size * model.xy_size * z
                    print("{:<32}".format(words[index]), end="")
                print()
            print()

    print(model.score(stories_correct, 5).mean())
    print_emissions()
    last_parameters = None
    last_score = torch.tensor(0).log()
    for epoch in range(1000):
        print(epoch)
        parameters = tuple(model.baum_welch_updates(stories_correct))
        model.set_paramters(*parameters)
        score = model.score(stories_correct, 5).mean()
        print(score)
        # print_emissions()
        # if (score - last_score).abs() < 1e-8:
        if score < last_score:
            print(f"stopped at step {epoch}")
            model.set_paramters(*last_parameters)
            model.score(stories_correct, 5)  # recalculate emissions
            break
        last_parameters = parameters
        last_score = score

    print(score_prediction_batch(model))
    print_emissions()
