import torchtext

# from datasets.lm import PennTreebank

from loadCloze import test
from main import main

result = main()

# text_field = torchtext.data.Field(batch_first=True)
#
# train, valid, test = PennTreebank.splits(text_field, newline_eos=True)
#
# text_field.build_vocab(train)
# text_vocab = text_field.vocab
