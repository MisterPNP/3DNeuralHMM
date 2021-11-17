import torch

from load_story_cloze import load_cloze_test
from neural_3d_hmm_model import Neural3DHMM
from scalar_3d_hmm_model import Scalar3DHMM

# TODO: Remove low-frequency tokens?

def score_prediction_single_story(prediction_model, first_index, last_index):
    # input test.json
    test_stories = load_cloze_test()[first_index:(last_index+1)]

    # get true and false stories
    true_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 4])]
    false_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 5])]

    # score true and false stories
    score_true = prediction_model.score(true_stories, 5)
    score_false = prediction_model.score(false_stories, 5)

    print("prediction scoring done.")
    # return a boolean to check if accurate prediction
    is_prediction_accurate = score_true > score_false
    return is_prediction_accurate


def score_prediction_batch(first_index, last_index):
    print("number of stories: " + str(last_index - first_index + 1))
    count = score_prediction_single_story(model, first_index, last_index)
    return count.float().mean()


stories = load_cloze_test()
print(stories.shape)

batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(64)
print(len(batches), batches[0].shape)

false_batch = stories[:64, torch.tensor([0, 1, 2, 3, 5])]

# model = Scalar3DHMM(2, 1, 5558)
# model = Scalar3DHMM(6, 6, 5558)
model = Neural3DHMM(6, 6, 5558)

# SGD
learning_rate = 0.0001
for epoch in range(3):
    for idx, batch in enumerate(batches):
        print(idx)
        p = model.score(batch, 5) - model.emission_log_p(batch[:, -1]).logsumexp(-1)
        p.sum(-1).backward()

        with torch.no_grad():
            print(p.mean())
            print((model.score(false_batch, 5) - model.emission_log_p(false_batch[:, -1]).logsumexp(-1)).mean())

            for parameter in model.parameters():
                # print(parameter.grad.norm())
                parameter += learning_rate * parameter.grad
                parameter.grad.zero_()

print("DONE LEARNING")

print(score_prediction_batch(0, 1000))

# TODO backward algorithm, instead of SGD
# stories_correct = stories[:, torch.tensor([0, 1, 2, 3, 4])]
# with torch.no_grad():
#     for epoch in range(1):
#         priors, transitions, emissions = model.baum_welch_updates(stories_correct)
#         print(priors, transitions, emissions)
