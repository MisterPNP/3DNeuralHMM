
import torch

from load_story_cloze import load_cloze_test


def score_prediction_single_story(prediction_model, first_index, last_index):
    test_stories = load_cloze_valid(filename)[first_index:(last_index+1)]

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


def score_prediction_batch(model, first_index, last_index):
    print("number of stories: " + str(last_index - first_index + 1))
    count = score_prediction_single_story(model, first_index, last_index)
    return count.float().mean()
