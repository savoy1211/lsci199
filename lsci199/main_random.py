from h_wordorder import *

def each_text(texts, text_names, n):
	for i in range(len(texts)):
		shuffled_text = list(texts[i])
		shuffle(shuffled_text)
		model = AdditiveSmoothingNGramModel(''.join(shuffled_text), n=n, add_tags=True)
		h_words, h_wordset = [], []
		for j in range(1,6):
			h_words_current, h_wordset_current = survey_text(model, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv(str(n)+"gram_model_results_windows1to5_"+str(text_names[i])+"_randomized2")
		print("Done! Created "+str(n)+"gram_model_results_windows1to5_"+str(text_names[i])+"_randomized2")

each_text([pride_and_prejudice,moby_dick,hard_times,two_cities],['pride_and_prejudice','moby_dick','hard_times','two_cities'], 2)

# each_text([pride_and_prejudice],['pride_and_prejudice'], 4)
# each_text([moby_dick],['moby_dick'], 4)
# each_text([hard_times],['hard_times'], 4)
# each_text([two_cities],['two_cities'], 3)
