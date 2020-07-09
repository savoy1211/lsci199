from h_wordorder import *

class NGramProbs:
  def __init__(self, ngram, index, count, total_count, vocabulary, alpha=0,):
    self.ngram = ngram
    self.index = index
    self.count = count + alpha
    self.vocabulary = vocabulary
    self.total_count = total_count + (vocabulary*alpha)
    self.alpha = alpha
    self.prob = count / total_count

  def print_data(self):
    print("   ngram", self.ngram)
    print("   index", self.index)
    print("   count", self.count)
    print("   total_count", self.total_count)
    print("   vocabulary", self.vocabulary)
    print("   alpha", self.alpha)
    print("   prob", self.prob)
    print()

class WordBank:
  def __init__(self, tokens, alpha, n=1):
    self.tokens = tokens
    self.n = n
    self.alpha = alpha
    self.dict_sizes, self.dict_vocabularies = self.get_dict_sizes()
    self.dict_wordbank = self.init_index()

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
    values = [NGramProbs(ngram=k[i], index=k[i][len(k[i])-1:len(k[i])], count=v[i], total_count=self.dict_sizes[len(list(k[i]))], vocabulary=self.dict_vocabularies[len(list(k[i]))], alpha=self.alpha) for i in range(len(k))]
    keys = [k[i] for i in range(len(k))] 
    return dict(zip(keys,values))

  def get_prob(self, tokens):
    index = tokens[len(tokens)-1:len(tokens)] # get last element, i.e. index
    try:
      prob_list = self.dict_wordbank[tuple(index)]
      return prob_list[tuple(tokens)].prob
    except KeyError:
      return self.alpha / (self.dict_sizes[len(tokens)]+self.dict_vocabularies[len(tokens)]*self.alpha)

  def get_dict_sizes(self):
    keys = [i+1 for i in range(self.n)]
    max_sum = sum(Counter(self.ngrams(self.tokens, 1)).values())
    values = [i+self.alpha for i in range(max_sum+1)]
    values.reverse()
    return dict(zip(keys, values[:self.n])), dict(zip(keys, [len(Counter(self.ngrams(self.tokens, i)).keys()) for i in range(1,self.n+1)]))

  def ngrams(self, tokens, n):
    """Get ngram counts for input"""
    to_remove = []
    all_tuples = []
    for i in range(len(tokens)-n+1):
      p = tuple(tokens[i:i+n])
      all_tuples.append(p)
    return all_tuples 

class NGramModel:
  def __init__(self, text, alpha, n=2, randomize_text=False, is_logprob=True):
    self.text = text
    self.alpha = alpha
    self.n = n
    self.randomize_text = randomize_text
    self.tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
    self.word_probs = WordBank(self.tokens, alpha=self.alpha, n=self.n)
    self.is_logprob = is_logprob

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
        # print(tokens[:i+1],"-",tokens[:i])
      for i in range(len(tokens) - self.n+1):
        total_prob *= w.get_prob(tokens[i:i+self.n]) / w.get_prob(tokens[i:i+self.n-1])
        # print(tokens[i:i+self.n],'-',tokens[i:i+self.n-1])
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
    return total_prob


  def logprob(self, prob):
    return math.log(prob,2.0)

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

def get_windows_within_sentence(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = []
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(model.text.strip(), realign_boundaries=False)
  for sentence in sentences:    
    sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
    window = []
    append_number, sentence_trunc = 0, sentence
    num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
    while append_number != num_windows(len(sentence), window_size):
      window = sentence_trunc[:window_size]
      if len(window) > 0 and len(window) == window_size:
        windows.append(window)
      append_number += 1
      sentence_trunc = sentence_trunc[1:]
  return windows

def get_windows_past_sentence(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows, window = [], []
  t = model.tokens
  append_number, sentence_trunc = 0, t
  num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
  while append_number != num_windows(len(t), window_size):
    window = sentence_trunc[:window_size]
    if len(window) > 0 and len(window) == window_size:
      windows.append(window)
    append_number += 1
    sentence_trunc = sentence_trunc[1:]
  return windows

def get_windows(model, window_size):
  if model.randomize_text is True:
    windows = get_windows_past_sentence(model, window_size)
  else:
    windows = get_windows_within_sentence(model, window_size)
  return windows

def survey_text(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = get_windows(model, window_size)
  logps_words = np.array([logp_words(model, window) for window in windows])
  logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
  H_words = np.mean(logps_words)
  H_word_sets = np.mean(logps_word_sets)
  if model.is_logprob is True:
    H_words, H_word_sets = -H_words, -H_word_sets
  return H_words, H_word_sets
