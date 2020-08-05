from get_probs import *

model = NgramModel(ModulateText(mega_text_90).tokens, alpha=0.1, n=4)
test = ModulateText(mega_text_10)
t = TestModel(model, test)