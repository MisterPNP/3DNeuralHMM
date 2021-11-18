
import torch

from load_story_cloze import load_cloze_test


def score_prediction_batch(model, first=0, last=-2):
    test_stories = load_cloze_valid()[first:(last + 1)]
    print("number of stories: " + len(test_stories))

    # get true and false stories
    true_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 4])]
    false_stories = test_stories[:, torch.tensor([0, 1, 2, 3, 5])]

    # score true and false stories
    score_true = prediction_model.score(true_stories, 5)
    score_false = prediction_model.score(false_stories, 5)

    print("prediction scoring done.")
    # return a boolean to check if accurate prediction
    is_prediction_accurate = score_true > score_false

    return is_prediction_accurate.float().mean()
