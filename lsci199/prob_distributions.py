from ngram_model import *

def get_prob_distributions(model):
	w = model.word_probs.dict_wordbank
	parameter = ""
	if model.randomize_text is False:
		parameter += "ordered_"
	else:
		parameter += "random"
	if model.sentence_inbound is True:
		parameter += "inbound"
	else:
		parameter += "outbound"
	for x in range(1, model.n+1):
		probs = []
		# bigram_probs = []
		# trigram_probs = []

		for i,j in w.items():
			for k,l in j.items():
				if len(l.ngram) == x:
					probs.append(l.prob)

		df = pd.DataFrame()
		df[str(x)+'grams'] = probs
		pd.DataFrame(df).to_csv("probs_"+str(x)+"grams_"+parameter+"_alpha0.5_mpht")
		print("Done! Created probs_"+str(x)+"grams_"+parameter+"_alpha0.5_mpht")

		# df = pd.DataFrame()
		# df['2grams'] = bigram_probs
		# pd.DataFrame(df).to_csv("probs_2grams_ordered_inbound_alpha0.5_mpht")
		# print("Done! Created probs_2grams_ordered_inbound_alpha0.5_mpht")

		# df = pd.DataFrame()
		# df['3grams'] = trigram_probs
		# pd.DataFrame(df).to_csv("probs_3grams_ordered_inbound_alpha0.5_mpht")
		# print("Done! Created probs_3grams_ordered_inbound_alpha0.5_mpht")


text = pride_and_prejudice+moby_dick+hard_times+two_cities
# get_prob_distributions(NGramModel(text, alpha=0.5, n=3, randomize_text=False, sentence_inbound=True))
# get_prob_distributions(NGramModel(text, alpha=0.5, n=3, randomize_text=False, sentence_inbound=False))
get_prob_distributions(NGramModel(text, alpha=0.5, n=3, randomize_text=True, sentence_inbound=True))
get_prob_distributions(NGramModel(text, alpha=0.5, n=3, randomize_text=True, sentence_inbound=False))

