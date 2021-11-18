import numpy as np

from load_story_cloze import *
from train import train
from score_cloze_accuracy import score_prediction_batch
from gradient_3d_hmm_model import Gradient3DHMM
from neural_3d_hmm_model import Neural3DHMM


get_processed_data() # YOU CAN COMMENT THIS OUT AFTER THE FIRST RUN!!!

index2word = pickle.load(open("../data/index2word.pkl", 'rb'))

stories = load_roc_test()

batch_size = 1000
batches = stories[:, torch.tensor([0, 1, 2, 3, 4])].split(batch_size)


# model = Gradient3DHMM(6, 6, len(index2word))
# learning_rate = 1e-2
# num_epochs = 1

model = Neural3DHMM(6, 6, len(index2word), token_embeddings=torch.load("../data/word_tensors.tensor"))
learning_rate = 3e-6
num_epochs = 1


analysis = train(model, batches, lr=learning_rate, num_epochs=num_epochs,
                 valid_batches=[load_cloze_valid()[:, :5]], accuracy_function=None)

print()
print("DONE LEARNING")

print()
print("TEST_LOSS", analysis['test_loss'])
print("VALID_LOSS", analysis['valid_loss'])
print("ACCURACY", score_prediction_batch(model))

model.compute_emission_matrix()
emission_matrix = model.emission_matrix.view(model.z_size, model.xy_size, model.xy_size, model.num_tokens)
highest = emission_matrix.topk(3, dim=-1).indices.numpy()
words = np.apply_along_axis(lambda r: ",".join(map(index2word.get, r)), -1, highest)
print("EMISSIONS")
for i, layer in enumerate(words):
    print(f"(layer {i})")
    for row in layer:
        for col in row:
            print(col, end="\t")
        print()
