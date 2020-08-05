from h_wordorder import *

class WordBank:
  def __init__(self, tokens, alpha, n=1):
    self.tokens = tokens
    self.n = n
    self.alpha = alpha
    self.dict_sizes, self.dict_vocabularies = self.get_dict_sizes()
    self.dict_wordbank = self.init_index()
    self.index_NGramConditionalProbs()
    
  def index_NGramConditionalProbs(self):
    w = self.dict_wordbank
    for i, j in w.items():
      for k,v in j.items():
        v.dict_wordbank = w
        v.init_total_count()
        v.init_prob()
  
  def init_index(self):
    current_n = 1
    dict0, final_dict = {}, {}
    while self.n - current_n >= 0:
      c = self.ngrams(self.tokens, current_n)
      for i in c:
        try:
          dict0[i[len(i)-1:len(i)]].append(i) 
        except KeyError:
          dict0[i[len(i)-1:len(i)]] = [i] 
      current_n += 1
    for k,v in dict0.items():
      final_dict[k] = self.get_ngram_probs(Counter(v))
    return final_dict

  def get_ngram_probs(self, counter):
    k,v = list(counter.keys()), list(counter.values())
    values = [NGramConditionalProbs(ngram=k[i], index=k[i][len(k[i])-1:len(k[i])], count=v[i], vocabulary=self.dict_vocabularies[len(list(k[i]))], alpha=self.alpha) for i in range(len(k))]
    keys = [k[i] for i in range(len(k))] 
    return dict(zip(keys,values))

  def get_prob(self, tokens):
    index = tokens[len(tokens)-1:len(tokens)] # get last element, i.e. index
    try:
      prob_list = self.dict_wordbank[tuple(index)]
      return prob_list[tuple(tokens)].prob
    except Exception:
      return self.alpha / (self.dict_sizes[len(tokens)]+self.dict_vocabularies[len(tokens)]*self.alpha)

  def get_dict_sizes(self):
    keys = [i+1 for i in range(self.n)]
    max_sum = sum(Counter(self.ngrams(self.tokens, 1)).values())
    values = [i+self.alpha for i in range(max_sum+1)]
    values.reverse()
    return dict(zip(keys, values[:self.n])), dict(zip(keys, [len(Counter(self.ngrams(self.tokens, i)).keys()) for i in range(1,self.n+1)]))

  def ngrams(self, tokens, n):
    """Get ngram counts for input"""
    all_tuples = []
    for i in range(len(tokens)-n+1):
      p = tuple(tokens[i:i+n])
      all_tuples.append(p)
    return all_tuples 

class NGramConditionalProbs:
  def __init__(self, ngram, index, count, vocabulary, alpha=0):
    self.ngram = ngram
    self.index = index
    self.count = count + alpha
    self.count_pre_alpha = count
    self.total_count = 0
    self.alpha = alpha
    self.prob = 0
    self.dict_wordbank = {}
    self.vocabulary = vocabulary
    self.unigram_count = 0
    
  def init_prob(self):
    self.prob = self.count / self.total_count
  
  def init_total_count(self):
    ngram_count = len(self.ngram)-1
    prefix_gram = self.ngram[:len(self.ngram)-1]
    index = tuple(prefix_gram[len(prefix_gram)-1:len(prefix_gram)])
    if ngram_count == 0:
      ngram_count = 1
      if self.unigram_count == 0:
        self.unigram_count = self.get_unigram_counts()
      self.total_count = self.unigram_count + (self.get_unigram_counts() * self.alpha)
    else:
      count = self.dict_wordbank[index][prefix_gram].count_pre_alpha
      self.total_count = count + (self.get_unigram_counts() * self.alpha)

  def get_unigram_vocabulary(self):
    w = self.dict_wordbank
    return len(w.items())

  def get_unigram_counts(self):
    w = self.dict_wordbank
    tot_counts = 0
    for i,j in w.items():
      index = i
      tot_counts += w[index][index].count_pre_alpha
    return tot_counts

  def print_data(self):
    print("   ngram", self.ngram)
    print("   index", self.index)
    print("   count", self.count)
    print("   total_count", self.total_count)
    print("   vocabulary", self.vocabulary)
    print("   alpha", self.alpha)
    print("   prob", self.prob)
    print()

class NGramModel:
  def __init__(self, text, alpha, n=2, randomize_text=False, randomize_sentence_inbound=False, randomize_n=1, sentence_inbound=True, include_smaller_windows=False, is_logprob=True, random_div=2000, unkify_most_common=False, replace_max_occurences=0, ordered_windows=True):
    self.text = text
    self.text_randomized = ''
    self.alpha = alpha
    self.n = n
    self.randomize_text = randomize_text
    self.randomize_sentence_inbound = randomize_sentence_inbound
    self.randomize_n = randomize_n
    self.include_smaller_windows = include_smaller_windows
    self.sentence_inbound = sentence_inbound
    self.random_div = random_div
    self.tokens_pre_randomized_text = self.init_tokens_pre_randomized_text()
    self.replace_max_occurences = replace_max_occurences
    self.unkify_most_common = unkify_most_common
    self.tokens = self.init_tokens()
    self.word_probs = WordBank(self.tokens, alpha=self.alpha, n=self.n)
    self.is_logprob = is_logprob
    self.ordered_windows = ordered_windows
  
  def init_tokens(self):
    if self.randomize_text is True:
      return list(filter((".").__ne__, self.tokens_pre_randomized_text))
    elif self.unkify_most_common is True:
      pre_unk = [token.casefold() for token in nltk.tokenize.word_tokenize(self.text) if token.isalnum()]
      unked = pre_unk
      most_common = Counter(pre_unk).most_common()[:self.replace_max_occurences]
      for i in most_common:
        unked = ["<<!!UNK!!>>" if x==str(i[0]) else x for x in unked]
      return unked
    return [token.casefold() for token in nltk.tokenize.word_tokenize(self.text) if token.isalnum()]
  
  def init_tokens_pre_randomized_text(self):
    if self.randomize_text is True:
      tokens_pre_randomized_text = [token.casefold() for token in nltk.tokenize.word_tokenize(self.text) if token.isalnum()]
      sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
      sentences = sent_detector.tokenize(self.text.strip(), realign_boundaries=False)
      # if self.randomize_sentence_inbound is False:
      #   for i in range(len(sentences)):
      #     tokens_pre_randomized_text.append('.')
      #   shuffle(tokens_pre_randomized_text)
      #   tokens_pre_randomized_text = self.divide_and_conquer(tokens_pre_randomized_text, self.random_div)
      #   self.text_randomized = " ".join(tokens_pre_randomized_text)
      #   return tokens_pre_randomized_text
      # elif self.sentence_inbound is False:
      if self.randomize_sentence_inbound is True:
        final_randomized_tokens = []
        for sentence in sentences:
          sentence_tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
          shuffle(sentence_tokens)
          final_randomized_tokens += sentence_tokens+['.']
        self.text_randomized = " ".join(final_randomized_tokens)
        return final_randomized_tokens
      else:
        shuffle(tokens_pre_randomized_text)
        self.text_randomized = " ".join(tokens_pre_randomized_text)
        return tokens_pre_randomized_text
      
  def total_prob(self, tokens):
    total_prob = 0
    w = self.word_probs
    total_prob = w.get_prob(tokens[:1])
    if self.n > len(tokens):
      for i in range(1, len(tokens)):
        total_prob *= w.get_prob(tokens[:i+1]) / w.get_prob(tokens[:i])
    elif self.n <= len(tokens):
      for i in range(1,self.n-1):
        total_prob *= w.get_prob(tokens[:i+1]) / w.get_prob(tokens[:i])
        print(tokens[:i+1],"/",tokens[:i])
      for i in range(len(tokens) - self.n+1):
        total_prob *= w.get_prob(tokens[i:i+self.n]) / w.get_prob(tokens[i:i+self.n-1])
        print(tokens[i:i+self.n],'/',tokens[i:i+self.n-1])
    return total_prob

  def total_logprob(self, tokens):
    total_prob = 0
    w = self.word_probs
    total_prob = self.logprob(w.get_prob(tokens[:1]))
    if self.n > len(tokens):
      for i in range(1, len(tokens)):
        total_prob += self.logprob(w.get_prob(tokens[:i+1])) - self.logprob(w.get_prob(tokens[:i]))
    elif self.n <= len(tokens):
      for i in range(1,self.n-1):
        total_prob += self.logprob(w.get_prob(tokens[:i+1])) - self.logprob(w.get_prob(tokens[:i]))
        # print(tokens[:i+1],"-",tokens[:i])
      for i in range(len(tokens) - self.n+1):
        total_prob += self.logprob(w.get_prob(tokens[i:i+self.n])) - self.logprob(w.get_prob(tokens[i:i+self.n-1]))
        # print(tokens[i:i+self.n],'-',tokens[i:i+self.n-1])
    if math.isnan(total_prob):
      return -float("inf")
    return total_prob

  def logprob(self, prob):
    if prob == 0:
      return -float("inf")
    return math.log(prob,2.0)

class TestCorpus(NGramModel):
  def __init__(self, text, sentence_inbound=True, randomize_text=False, randomize_sentence_inbound=False, include_smaller_windows=False, unkify_most_common=False, replace_max_occurences=0, ordered_windows=True):
    self.text = text
    self.text_randomized = ''
    self.randomize_text = randomize_text
    self.randomize_sentence_inbound = randomize_sentence_inbound
    self.tokens_pre_randomized_text = self.init_tokens_pre_randomized_text()
    self.unkify_most_common = unkify_most_common
    self.replace_max_occurences = replace_max_occurences    
    self.tokens = self.init_tokens()
    self.sentence_inbound = sentence_inbound
    self.include_smaller_windows = include_smaller_windows
    self.ordered_windows = ordered_windows


def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
    if model.is_logprob is True:
      return model.total_logprob(tokens)
    return model.total_prob(tokens)

def logp_word_set(model, tokens): 
    """Get plog sum of each tokens' permutations"""
    logprobs = [logp_words(model, reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
    if not logprobs:
        logprobs = [0]
    return scipy.special.logsumexp(logprobs)

def get_windows_sentence_inbound(test, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = []
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  if test.ordered_windows is True:
    sentences = sent_detector.tokenize(test.text.strip(), realign_boundaries=False)
  elif test.ordered_windows is False:
    sentences = sent_detector.tokenize(" ".join(test.tokens_pre_randomized_text), realign_boundaries=False) 
  for sentence in sentences:    
    sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
    window = []
    append_number, sentence_trunc = 0, sentence
    num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
    while append_number != num_windows(len(sentence), window_size):
      window = sentence_trunc[:window_size]
      if test.include_smaller_windows is False:
        if len(window) > 0 and len(window) == window_size:
          windows.append(window)
      else:
        if len(window) > 0 and len(window) <= window_size:
          windows.append(window)
      append_number += 1
      sentence_trunc = sentence_trunc[1:]
  return windows

def get_windows_sentence_outbound(test, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows, window = [], []
  t = test.tokens
  append_number, sentence_trunc = 0, t
  num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
  while append_number != num_windows(len(t), window_size):
    window = sentence_trunc[:window_size]
    if len(window) > 0 and len(window) == window_size:
      windows.append(window)
    append_number += 1
    sentence_trunc = sentence_trunc[1:]
  return windows

def get_windows(test, window_size):
  if test.sentence_inbound is True:
    windows = get_windows_sentence_inbound(test, window_size)
  else:
    windows = get_windows_sentence_outbound(test, window_size)
  return windows

def survey_text(model, test, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = get_windows(model, window_size)
  logps_words = np.array([logp_words(model, window) for window in windows])
  logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
  zero_equivalent = logp_word_set(model, window_size*["<<!!ZERO!!>>>"])
  ratio_of_zeros_permuted_windows = logps_word_sets.tolist().count(zero_equivalent) / len(logps_word_sets)
  H_words = np.mean(logps_words)
  H_word_sets = np.mean(logps_word_sets)
  if model.is_logprob is True:
    H_words, H_word_sets = -H_words, -H_word_sets
  return H_words, H_word_sets, ratio_of_zeros_permuted_windows
