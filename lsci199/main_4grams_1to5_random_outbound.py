from h_wordorder import *
from ngram_model import *

def each_text(texts, text_names, n, alpha=0):
	for i in range(len(texts)):
		model = NGramModel(texts[i], alpha=alpha, n=n)
		h_words, h_wordset = [], []
		for j in range(1,6):
			h_words_current, h_wordset_current = survey_text(model, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv(str(n)+"gram_alpha"+str(alpha)+"_1to5_random_outbound_"+str(text_names[i]))
		print("Done! Created "+str(n)+"gram_alpha"+str(alpha)+"_1to5_random_outbound_"+str(text_names[i]))

each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 4, alpha=0)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 4, alpha=0.5)
each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 4, alpha=1)
