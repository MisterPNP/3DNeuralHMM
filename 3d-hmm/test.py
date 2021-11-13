import torchtext
import pprint

# from datasets.lm import PennTreebank

from load_story_cloze import test

result = test()
#print(result)

# text_field = torchtext.data.Field(batch_first=True)
#
# train, valid, test = PennTreebank.splits(text_field, newline_eos=True)
#
# text_field.build_vocab(train)
# text_vocab = text_field.vocab
