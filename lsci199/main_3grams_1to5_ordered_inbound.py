from h_wordorder import *
from ngram_model import *
from download_local_gutenberg_texts import *
import time

def each_text(training_text, training_text_names, testing_text, testing_text_names, n, alpha=0):
	for i in range(len(training_text)):
		start = time.time()
		model = NGramModel(training_text[0], alpha=alpha, n=n)
		test = TestCorpus(testing_text[0])
		h_words, h_wordset, zeros_permutations = [], [], []
		for j in range(1,6):
			h_words_current, h_wordset_current, zeros_permutations_current = survey_text(model, test, j)
			print("h_words",h_words_current, "h_wordset", h_wordset_current, "zeros_permutations", zeros_permutations_current)
			h_words.append(h_words_current)
			h_wordset.append(h_wordset_current)
			zeros_permutations.append(zeros_permutations_current)

		d = { 'h_words': h_words, 'h_wordset': h_wordset, 'zeros_permutations': zeros_permutations}
		df = pd.DataFrame(data=d, dtype=np.float64)
		pd.DataFrame(df).to_csv(str(n)+"gram_ordered_inbound_alpha"+str(alpha)+"_1to5_"+str(training_text_names[0])+str(testing_text_names[0]))
		print("Done! Created "+str(n)+"gram_ordered_inbound_alpha"+str(alpha)+"_1to5_"+str(training_text_names[0])+str(testing_text_names[0]))
		end = time.time()
		print(end-start)

# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.00)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.01)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.10)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.25)
# each_text([pride_and_prejudice+moby_dick+hard_times+two_cities],['mpht'], 3, alpha=0.50)

# 3grams_train(90)_test(10)
each_text([mega_text_90],['train(mega_text_90)_'], [mega_text_10], ['_test(mega_text_10)'], 3, alpha=0.10)

# 2grams_train(90)_test(10)
# each_text([mega_text_90],['train(mega_text_90)_'], [mega_text_10], ['test(mega_text_10)'], 2, alpha=0.10)
