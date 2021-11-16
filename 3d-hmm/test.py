import torch

from load_story_cloze import load_cloze_test
from scalar_3d_hmm_model import Scalar3DHMM


def score_prediction_single_story(model, story_index):
    # input test.json
    test_stories = load_cloze_test()
    story = test_stories[story_index]

    # get true and false stories
    true_story = story[(1, 2, 3, 4, 5)]
    false_story = story[(1, 2, 3, 4, 6)]

    # score true and false stories
    score_true = model.score(true_story, 5)
    score_false = model.score(false_story, 5)

    print("score for true ending: " + str(score_true))
    print("score for false ending: " + str(score_false))

    # return a boolean to check if accurate prediction
    is_prediction_accurate = score_true > score_false
    print("accurate prediction: " + str(is_prediction_accurate))

    return is_prediction_accurate


stories = load_cloze_test()
print(stories.shape)

batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(64)
print(len(batches), batches[0].shape)

false_batch = stories[:64, torch.tensor([0, 1, 2, 3, 5])]

# model = Scalar3DHMM(2, 1, 5558)
model = Scalar3DHMM(6, 6, 5558)

# SGD
lr = 0.001
for epoch in range(1):
    for idx, batch in enumerate(batches):
        print(idx)
        p = model.score(batch, 5)
        p.sum(-1).backward()
        with torch.no_grad():
            print(p.mean())
            print(model.score(false_batch, 5).mean())

            model.emission_matrix_unnormalized += lr * model.emission_matrix_unnormalized.grad
            model.emission_matrix_unnormalized.grad.zero_()
            model.transition_matrix_unnormalized += lr * model.transition_matrix_unnormalized.grad
            model.transition_matrix_unnormalized.grad.zero_()
            model.state_priors_unnormalized += lr * model.state_priors_unnormalized.grad
            model.state_priors_unnormalized.grad.zero_()
print("DONE LEARNING")

score_prediction_single_story(model, 0)

# TODO backward algorithm, instead of SGD
# ...
