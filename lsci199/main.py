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
		model = AdditiveSmoothingNGramModel(texts[i], n=2, add_tags=True)
		h_words, h_wordset = [], []
		for j in range(1,6):
			h_words_current, h_wordset_current = survey_text(model, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv("bigram_model_results_windows1to5_"+str(text_names[i]))
		print("Done! Created bigram_model_results_windows1to5_"+str(text_names[i]))

def increasing_texts(texts, window_size, text_name):
	text = ""
	h_words, h_wordset = [], []
	for i in range(len(texts)):
		text += texts[i]
		model = AdditiveSmoothingNGramModel(text, n=3)
		h_words_current, h_wordset_current = survey_text(model, window_size)
		print("h_words",h_words_current, "h_wordset", h_wordset_current)
		h_words.append(h_words_current)
		h_wordset.append(h_wordset_current)
	d = { 'h_words': h_words, 'h_wordset': h_wordset}
	df = pd.DataFrame(data=d, dtype=np.float64)
	pd.DataFrame(df).to_csv("bigram_model_results_window"+str(window_size)+"_"+text_name)
	print("Done! Created bigram_model_results_window"+str(window_size)+"_"+text_name)

each_text([pride_and_prejudice[:100000]], ['pride_and_prejudice'])

# text = pride_and_prejudice
# text_name = "pride_and_prejudice"
# increasing_texts([text[:50000], text[:100000], text[:150000], text[:200000], text[:250000]],1,text_name)
# increasing_texts([text[:50000], text[:100000], text[:150000], text[:200000], text[:250000]],2,text_name)
# increasing_texts([text[:50000], text[:100000], text[:150000], text[:200000], text[:250000]],3,text_name)
# increasing_texts([text[:50000], text[:100000], text[:150000], text[:200000], text[:250000]],4,text_name)
