import torchtext

from datasets.lm import PennTreebank

text_field = torchtext.data.Field(batch_first=True)

train, valid, test = PennTreebank.splits(text_field, newline_eos=True)

text_field.build_vocab(train)
text_vocab = text_field.vocab

print(text_vocab)
print("finished")
