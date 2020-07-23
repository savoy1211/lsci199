from h_wordorder import *
from ngram_model import *
import time

def each_text(texts, text_names, n, alpha=0, random_div=2000):
	# start = time.time()
	# model = NGramModel(texts[0], alpha=alpha, n=n)
	# h_words, h_wordset = [], []
	# for j in range(1,6):
	# 	h_words_current, h_wordset_current = survey_text(model, j)
	# 	print("h_words",h_words_current, "h_wordset", h_wordset_current)
	# 	h_words.append(h_words_current)
	# 	h_wordset.append(h_wordset_current)

	# d = { 'h_words': h_words, 'h_wordset': h_wordset}
	# df = pd.DataFrame(data=d, dtype=np.float64)
	# pd.DataFrame(df).to_csv(str(n)+"gram_ordered_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# print("Done! Created "+str(n)+"gram_ordered_text_ordered_window_inbound_alpha"+str(alpha)+"_1to5_"+str(text_names[0]))
	# end = time.time()
	# print(end-start)
	
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
	model = NGramModel(texts[0], alpha=alpha, n=n, randomize_text=True, random_div=random_div, ordered_windows=False)
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

each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht_1'], 3, alpha=0.10, random_div=2000)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht_2'], 3, alpha=0.10, random_div=2000)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht_3'], 3, alpha=0.10, random_div=2000)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht_4'], 3, alpha=0.10, random_div=2000)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht_5'], 3, alpha=0.10, random_div=2000)

# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.25)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.50)
