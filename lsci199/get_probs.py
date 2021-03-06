from modulate_text import *
from small_ngram_model import *
from load_wiki_txt import *
import numpy as np
import pandas as pd
import scipy.special
import itertools
import time

class TestModel:
  def __init__(self, trained_model, test_text):
    self.test_state = test_text.state
    self.trained_model = trained_model
    self.test_text = test_text
    
    start = time.time()
    h_words, h_wordset, zeros_permutations = [], [], []
    for i in range(1,6):
      h_words_current, h_wordset_current, zeros_permutations_current = self.survey_text(trained_model, test_text, i)
      print("h_words",h_words_current, "h_wordset", h_wordset_current, "zeros_permutations", zeros_permutations_current)
      h_words.append(h_words_current)
      h_wordset.append(h_wordset_current)
      zeros_permutations.append(zeros_permutations_current)

    d = { 'h_words': h_words, 'h_wordset': h_wordset, 'zeros_permutations': zeros_permutations}
    df = pd.DataFrame(data=d, dtype=np.float64)
    if self.test_state == "ordered":
      pd.DataFrame(df).to_csv(str(self.trained_model.n)+"gram_ordered_inbound_alpha"+str(self.trained_model.alpha)+"_1to5_90_10_TURKISH")
      print("Done! Created "+str(self.trained_model.n)+"gram_ordered_inbound_alpha"+str(self.trained_model.alpha)+"_1to5_90_10_TURKISH")
    else:
      pd.DataFrame(df).to_csv(str(self.trained_model.n)+"gram_random_inbound_alpha"+str(self.trained_model.alpha)+"_1to5_90_10_TURKISH")
      print("Done! Created "+str(self.trained_model.n)+"gram_random_inbound_alpha"+str(self.trained_model.alpha)+"_1to5_90_10_TURKISH")
    end = time.time()
    print(end-start)

  def logp_words(self, tokens):
      """Get conditional plog of words using ngram model"""
      return self.trained_model.window_logprob(tokens)
    
  def logp_word_set(self, tokens): 
      """Get plog sum of each tokens' permutations"""
      logprobs = [self.logp_words(reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
      if not logprobs:
          logprobs = [0]
      return scipy.special.logsumexp(logprobs)

  def get_windows_sentence_inbound(self, test, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(test.text.strip(), realign_boundaries=False)
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
  
  def get_windows_sentence_outbound(self, test, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows, window = [], []
    t = test.random_tokens
    append_number, sentence_trunc = 0, t
    num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
    while append_number != num_windows(len(t), window_size):
      window = sentence_trunc[:window_size]
      if len(window) > 0 and len(window) == window_size:
        windows.append(window)
      append_number += 1
      sentence_trunc = sentence_trunc[1:]
    return windows
  
  def get_windows(self, test, window_size):
    if test.state != "random across sentence":
      windows = self.get_windows_sentence_inbound(test, window_size)
    else:
      windows = self.get_windows_sentence_outbound(test, window_size)
    return windows
  
  def survey_text(self, model, test, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = self.get_windows(test, window_size)
    logps_words = np.array([self.logp_words(window) for window in windows])
    logps_word_sets = np.array([self.logp_word_set(window) for window in windows])
    zero_equivalent = self.logp_word_set(window_size*["<<!!ZERO!!>>>"])
    ratio_of_zeros_permuted_windows = logps_word_sets.tolist().count(zero_equivalent) / len(logps_word_sets)
    H_words = -np.mean(logps_words)
    H_word_sets = -np.mean(logps_word_sets)
    return H_words, H_word_sets, ratio_of_zeros_permuted_windows
