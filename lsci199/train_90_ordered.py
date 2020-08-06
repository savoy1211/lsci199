from get_probs import *

model = NgramModel(ModulateText(turkish_90).tokens, alpha=0.1, n=3)
test = ModulateText(mega_text_10)
t = TestModel(model, test)

# model = NgramModel(ModulateText(turkish_10).tokens, alpha=0.5, n=3)
# test = ModulateText(mega_text_10)
# t = TestModel(model, test)



model = NgramModel(ModulateText(mega_text_90, randomize_across_sentence=True).random_tokens, alpha=0.1, n=3)
test = ModulateText(mega_text_10, randomize_within_sentence=True)
t = TestModel(model, test)

# model = NgramModel(ModulateText(mega_text_90, randomize_across_sentence=True).random_tokens, alpha=0.5, n=3)
# test = ModulateText(mega_text_10, randomize_within_sentence=True)
# t = TestModel(model, test)