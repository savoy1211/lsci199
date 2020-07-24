from h_wordorder import *
from ngram_model import *
from download_gutenberg_texts import *
import time

def each_text(texts, text_names, n, alpha=0):
	start = time.time()
	model = NGramModel(texts[0], alpha=alpha, n=n)
	h_words, h_wordset = [], []
	for j in range(1,6):
		h_words_current, h_wordset_current = survey_text(model, j)
		print("h_words",h_words_current, "h_wordset", h_wordset_current)
		h_words.append(h_words_current)
		h_wordset.append(h_wordset_current)

	d = { 'h_words': h_words, 'h_wordset': h_wordset}
	df = pd.DataFrame(data=d, dtype=np.float64)
	pd.DataFrame(df).to_csv(str(n)+"gram_ordered_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	print("Done! Created "+str(n)+"gram_ordered_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	end = time.time()
	print(end-start)
	
	# start = time.time()
	# model = NGramModel(texts[0], alpha=alpha, n=n, ordered_windows=False)
	# h_words, h_wordset = [], []
	# for j in range(1,6):
	# 	h_words_current, h_wordset_current = survey_text(model, j)
	# 	print("h_words",h_words_current, "h_wordset", h_wordset_current)
	# 	h_words.append(h_words_current)
	# 	h_wordset.append(h_wordset_current)

	# d = { 'h_words': h_words, 'h_wordset': h_wordset}
	# df = pd.DataFrame(data=d, dtype=np.float64)
	# pd.DataFrame(df).to_csv(str(n)+"gram_ordered_text_random_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# print("Done! Created "+str(n)+"gram_ordered_text_random_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# end = time.time()
	# print(end-start)
	
	start = time.time()
	model = NGramModel(texts[0], alpha=alpha, n=n, randomize_text=True, ordered_windows=False)
	h_words, h_wordset = [], []
	for j in range(1,6):
		h_words_current, h_wordset_current = survey_text(model, j)
		print("h_words",h_words_current, "h_wordset", h_wordset_current)
		h_words.append(h_words_current)
		h_wordset.append(h_wordset_current)

	d = { 'h_words': h_words, 'h_wordset': h_wordset}
	df = pd.DataFrame(data=d, dtype=np.float64)
	pd.DataFrame(df).to_csv(str(n)+"gram_random_text_random_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	print("Done! Created "+str(n)+"gram_random_text_random_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	end = time.time()
	print(end-start)
	
	# start = time.time()
	# model = NGramModel(texts[0], alpha=alpha, n=n, randomize_text=True)
	# h_words, h_wordset = [], []
	# for j in range(1,6):
	# 	h_words_current, h_wordset_current = survey_text(model, j)
	# 	print("h_words",h_words_current, "h_wordset", h_wordset_current)
	# 	h_words.append(h_words_current)
	# 	h_wordset.append(h_wordset_current)

	# d = { 'h_words': h_words, 'h_wordset': h_wordset}
	# df = pd.DataFrame(data=d, dtype=np.float64)
	# pd.DataFrame(df).to_csv(str(n)+"gram_random_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# print("Done! Created "+str(n)+"gram_random_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# end = time.time()
	# print(end-start)

# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.10)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.25)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.50)
each_text([mega_text_50],['mega_text_50'], 3, alpha=0.10)
each_text([mega_text_100],['mega_text_100'], 3, alpha=0.10)
each_text([mega_text_50],['mega_text_50'], 3, alpha=0.01)
each_text([mega_text_100],['mega_text_100'], 3, alpha=0.01)