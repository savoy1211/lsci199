from h_wordorder import *

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

# nltk.download("punkt") # This gives us access to nltk.tokenize.word_tokenize
url_prefix = "http://www.socsci.uci.edu/~rfutrell/teaching/lsci109-w2020/data/"
pride_and_prejudice = download_gutenberg_text(url_prefix + "1342-0.txt")
two_cities = download_gutenberg_text(url_prefix + "98-0.txt")
moby_dick = download_gutenberg_text(url_prefix + "2701-0.txt")
hard_times = download_gutenberg_text(url_prefix + "786-0.txt")

def each_text(texts, text_names):
	for i in range(len(texts)):
		model = AdditiveSmoothingNGramModel(texts[i], n=4, add_tags=True)
		h_words, h_wordset = [], []
		for j in range(3,8):
			h_words_current, h_wordset_current = survey_text(model, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv("4gram_model_results_windows3to7_"+str(text_names[i]))
		print("Done! Created 4gram_model_results_windows3to7_"+str(text_names[i]))


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

each_text([pride_and_prejudice,moby_dick,hard_times,two_cities],['pride_and_prejudice','moby_dick','hard_times','two_cities'])
