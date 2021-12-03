
import torch

from load_story_cloze import load_cloze_valid


def score_prediction_batch(model, first=0, last=-2):
    test_stories = load_cloze_valid()[first:(last + 1)].to(next(model.parameters()).device)
    # print("number of stories:", len(test_stories))

    # get true and false stories
    true_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 4])]
    false_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 5])]

    # score true and false stories
    score_true = model.score(true_stories, 5) - model.emission_log_p(true_stories[:, -1]).logsumexp(-1)
    score_false = model.score(false_stories, 5) - model.emission_log_p(false_stories[:, -1]).logsumexp(-1)
    # print("prediction scoring done.")

    # return a boolean to check if accurate prediction
    is_prediction_accurate = score_true > score_false

    return is_prediction_accurate.float().mean()
