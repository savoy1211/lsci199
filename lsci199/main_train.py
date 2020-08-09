from lm_results import *
import pickle as p

# # Get Basline Results Ordered Inbound
# test = ModulateText(chinese_full, language='chinese by character')
# print("Test text created!")
# model = NgramModel(test.tokens, alpha=0, n=3)
# print("LM setup complete!")
# t = LMResultsBaseline(model, test, "n3_a0_oi_CHINESE_full")
# file = open("LM_n3_a0_oi_CHINESE_full", "wb")
# p.dump(t, file)
# print("LM created!")

# # file = open("LM_3gram_alpha0_(moby_dick)_ordered_FULL", "rb")
# # p.load(file)

# nltk.download('punkt')

# Get Basline Results Ordered Inbound
test = ModulateText(english_full, state="ordered")
print("Test text created!")
model = NgramModel(test.tokens, alpha=0.1, n=3)
print("LM setup complete!")
t = LMResults(model, test)
# t.get_results("LM_n3_a0.1_oi_(ENGLISH)_full.txt")
file = open("LM_n3_a0.1_oi_(ENGLISH)_full", "wb")
p.dump(t, file)
print("LM created!")

# file = open("LM_3gram_alpha0_(moby_dick)_ordered_FULL", "rb")
# p.load(file)



# Get Results Ordered Inbound

# model = NgramModel(ModulateText(english_train).tokens, alpha=0.1, n=3)
# test = ModulateText(english_test)
# t = LMResults(model, test, "n3_a0.1_oi_ENGLISH_full")

# file = open("LM_n3_a0.1_oi_ENGLISH_full", "wb")
# p.dump(t, file)

# # Get Results Random Inboun

# model = NgramModel(ModulateText(mega_text_90, randomize_across_sentence=True).random_tokens, alpha=0.1, n=3)
# test = ModulateText(mega_text_10, randomize_within_sentence=True)
# t = LMResults(model, test)

# model = NgramModel(ModulateText(mega_text_90, randomize_across_sentence=True).random_tokens, alpha=0.5, n=3)
# test = ModulateText(mega_text_10, randomize_within_sentence=True)
# t = TestModel(model, test)