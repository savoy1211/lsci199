from h_wordorder import *

def each_text(texts, text_names, n):
	for i in range(len(texts)):
		model = AdditiveSmoothingNGramModel(texts[i], n=n, add_tags=True)
		h_words, h_wordset = [], []
		for j in range(1,6):
			h_words_current, h_wordset_current = survey_text(model, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv(str(n)+"gram_model_results_windows1to5_"+str(text_names[i]))
		print("Done! Created "+str(n)+"gram_model_results_windows1to5_"+str(text_names[i]))


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model_ = GPT2LMHeadModel.from_pretrained('gpt2')
# model_.eval()


# def each_text(texts, text_names):
# 	for i in range(len(texts)):
# 		model = GPT2Model(model_, tokenizer, texts[i])
# 		h_words, h_wordset = [], []
# 		for j in range(1,6):
# 			h_words_current, h_wordset_current = survey_text_gpt2(model, j)
# 			print("h_words",h_words_current, "h_wordset", h_wordset_current)
# 			h_words.append(h_words_current)
# 			h_wordset.append(h_wordset_current)

# 		d = { 'h_words': h_words, 'h_wordset': h_wordset}
# 		df = pd.DataFrame(data=d, dtype=np.float64)
# 		pd.DataFrame(df).to_csv("gpt2_results_windows1to5_"+str(text_names[i]))
# 		print("Done! Created gpt2_results_windows1to5_"+str(text_names[i]))

each_text([pride_and_prejudice,moby_dick,hard_times,two_cities],['pride_and_prejudice','moby_dick','hard_times','two_cities'], 3)
