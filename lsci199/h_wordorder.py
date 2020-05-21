import io
import random
import sys
import requests
import pandas as pd
import numpy as np
import scipy.special
import nltk
import math
from collections import Counter
import itertools
nf = -float("inf")

class AdditiveSmoothingNGramModel:
    def __init__(self, text, alpha=1, n=2):
        self.text = text
        self.alpha = alpha
        self.n = n
        self.tokens = self.tokens_init(text)
        self.ngram_logprobs, self.prefix_logprobs = self.logprob_init(self.tokens)

    def tokens_init(self, text):
      tokens = []
      sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
      sentences = sent_detector.tokenize(text.strip(), realign_boundaries=False)
      for sentence in sentences:
        current_tokens = [token.casefold() for token in nltk.word_tokenize(sentence) if token.isalnum()]
        current_tokens.insert(0,"<<START>>")
        current_tokens.append("<<END>>")
        tokens += current_tokens
      return tokens

    def logprob_init(self, tokens):
      # p(w1, w2, ..., wn) = product from t=1 to t=n of p(w_t | w_{t - (n-1)} ..., w_{t - 1})
      ngrams = Counter(self.ngrams(tokens,self.n))
      total_val = sum(ngrams.values())
      ngram_probs = {gram: -math.log((value/total_val),2.0) for gram, value in ngrams.items()}

      prefix = Counter(self.ngrams(tokens, self.n-1))
      total_val = sum(prefix.values())
      prefix_probs = {gram: -math.log((value/total_val),2.0) for gram, value in prefix.items()}

      return ngram_probs, prefix_probs

    def total_prob(self, tokens):

      ngrams = self.ngrams(tokens, self.n)
      prefix = self.ngrams(tokens, self.n-1)
      logp_words = 0
      if len(tokens) == 1:
        try:
          logp_words = self.prefix_logprobs[prefix[0]]
        except Exception:
            return nf
      elif len(tokens) > 1:
        for i in range(0, len(tokens)-1):
            try:
              logp_words += float(self.ngram_logprobs[ngrams[i]] - self.prefix_logprobs[prefix[i]])
            except Exception:
              return nf
      return -logp_words

    def ngrams(self, text, n):
      """Get ngram counts for input"""
      return [tuple(text[i:i+n]) for i in range(len(text)-n+1) if text[i:i+n]]

def test_total_logprobs():
    # Test 1
    text = "foo bar foo bar foo"
    model = AdditiveSmoothingNGramModel(text)
    model_logp = logp_words(model,model.tokens)
    actual_logp = 3*(-math.log(7/6,2.0)) + 2*(-math.log(7/9,2.0)) + (-math.log(7/18,2.0))
    assert round(model_logp, 12) == round(actual_logp, 12)
     
    # Test 2
    # text = "foo bar foo bar foo"
    model_prob = logp_words(model,["foo","foo","bar"])
    assert round(model_prob, 12) == nf
    
    # Test 3
    # text = "foo bar foo bar foo"
    assert round(logp_words(model,["foo","foo"]), 12) == nf

    # Test 4
    text = "In the beginning, was the word, and the word, was with God, and the word was God."
    model = AdditiveSmoothingNGramModel(text)
    model_prob = logp_words(model,["the","word"])
    actual_prob = -math.log(3/18,2.0) + math.log(4/19,2.0) # plog((the, word) - the)
    assert round(model_prob, 12) == round(actual_prob, 12)
    
    # Test 5
    # text = "In the beginning, was the word, and the word, was with God, and the word was God."
    assert logp_words(model,["word"]) == -math.log(3/19,2.0) # plog(word)
    
    # Test 6
    # text = "In the beginning, was the word, and the word, was with God, and the word was God."
    assert logp_words(model,["haberdashery"]) == nf

    # Test 7 
    text = "foo. foo bar. bar bar foo foo bar. foo bar foo bar foo. bar foo."
    model = AdditiveSmoothingNGramModel(text)
    model_prob = logp_words(model, model.tokens)
    actual_prob = 4*(-math.log(4/24,2.0)+math.log(5/25,2.0)) + 4*(-math.log(4/24,2.0)+math.log(8/25,2.0)) + 4*(-math.log(4/24,2.0)+math.log(7/25,2.0)) + 3*(-math.log(3/24,2.0)+math.log(5/25,2.0)) + 3*(-math.log(3/24,2.0)+math.log(8/25,2.0)) + 2*(-math.log(2/24,2.0)+math.log(7/25,2.0)) + 2*(-math.log(2/24,2.0)+math.log(5/25,2.0)) + (-math.log(1/24,2.0)+math.log(7/25,2.0)) + (-math.log(1/24,2.0)+math.log(8/25,2.0))
    assert round(model_prob, 12) == round(actual_prob, 12)

def test_logp_word_set():
  # Test 1
  text = "1 2 3"
  model = AdditiveSmoothingNGramModel(text)
  h_word_sets_test = logp_word_set(model, ["1","2","3"])
  h_word_sets_actual = scipy.special.logsumexp([
    (-math.log(1/4,2.0) + math.log(1/5,2.0)) + (-math.log(1/4,2.0) + math.log(1/5,2.0)), # [1,2,3]
    -float("inf"), # [1,3,2]
    -float("inf"), # [2,1,3]
    -float("inf"), # [2,3,1]
    -float("inf"), # [3,1,2]
    -float("inf"), # [3,2,1]
  ])
  assert round(h_word_sets_test, 12) == round(h_word_sets_actual, 12)

  # Test 2 
  # text = "1 2 3"
  h_word_sets_test = logp_word_set(model, ["1","2","4"])
  assert h_word_sets_test == nf

  # Test 3
  text = "foo bar foo bar foo foo bar"
  model = AdditiveSmoothingNGramModel(text)
  h_word_sets_test = logp_word_set(model,["foo", "bar", "foo"])
  h_word_sets_actual = scipy.special.logsumexp([
    (-math.log(3/8,2.0)+math.log(4/9,2.0)) + (-math.log(2/8,2.0)+math.log(3/9,2.0)), # foo bar foo
    (-math.log(3/8,2.0)+math.log(4/9,2.0)) + (-math.log(2/8,2.0)+math.log(3/9,2.0)), # foo bar foo 
    (-math.log(1/8,2.0)+math.log(4/9,2.0)) + (-math.log(3/8,2.0)+math.log(4/9,2.0)), # foo foo bar
    (-math.log(1/8,2.0)+math.log(4/9,2.0)) + (-math.log(3/8,2.0)+math.log(4/9,2.0)), # foo foo bar
    (-math.log(2/8,2.0)+math.log(3/9,2.0)) + (-math.log(1/8,2.0)+math.log(4/9,2.0)), # bar foo foo
    (-math.log(2/8,2.0)+math.log(3/9,2.0)) + (-math.log(1/8,2.0)+math.log(4/9,2.0)), # bar foo foo
  ])
  assert round(h_word_sets_test, 12) == round(h_word_sets_actual, 12)

def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
    return model.total_prob(tokens)

def logp_word_set(model, tokens): 
    """Get plog sum of each tokens' permutations"""
    logprobs = [model.total_prob(reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
    if not logprobs:
        logprobs = [0]
    return scipy.special.logsumexp(logprobs)

def H_words(model, set_of_sequences):
    logprobs = np.array([logp_words(model, sequence) for sequence in set_of_sequences])
    return -(np.exp(logprobs) * logprobs).sum(axis=-1)

def H_word_sets(model, set_of_sequences):
    outer_logps = np.array([logp_words(model, sequence) for sequence in set_of_sequences])
    inner_logps = np.array([logp_word_set(model, sequence) for sequence in set_of_sequences])
    return -(np.exp(outer_logps)*inner_logps).sum(axis=-1)

def survey_text(model, tokens, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = []
    for token in tokens:
      window = []
      if len(tokens[:window_size]) == window_size: 
        window = tokens[:window_size]
        windows.append(window)
        tokens = tokens[1:]
    logps_words = np.array([logp_words(model, window) for window in windows])
    logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
    H_words = -np.mean(logps_words)
    H_word_sets = -np.mean(logps_word_sets)
    return (H_words, H_word_sets)

if __name__ == '__main__':
    import nose
    nose.runmodule()