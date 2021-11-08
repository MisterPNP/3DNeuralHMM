from torchtext import data
from data import Field

import datasets

text_field = data.Field(batch_first=True)

train, valid, test = datasets.lm.PennTreebank.splits(text_field, newline_eos=True)

text_field.build_vocab(train)
text_vocab = text_field.vocab

print(text_vocab)
