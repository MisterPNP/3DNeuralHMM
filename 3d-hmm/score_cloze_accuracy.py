
import torch

from load_story_cloze import load_cloze_test


def score_prediction_batch(model, first_index, last_index):
    print("number of stories: " + str(last_index - first_index + 1))
    count = score_prediction_single_story(model, first_index, last_index)
    return count.float().mean()
