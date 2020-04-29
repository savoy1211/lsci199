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

class AdditiveSmoothingNGramModel:
    def __init__(self, text, alpha=1, n=2):
        self.text = text
        self.alpha = alpha
        self.n = n
        self.tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(text)]
        self.ngram_logprobs = self.logprob(self.tokens)

    def logprob(self, tokens):
      n = self.n
      INF = float('inf')
      ngrams = Counter(self.ngrams(tokens, n))
      logZ_ngrams = math.log(sum(ngrams.values()), 2.0)
      ngram_probs = {gram: -1*(math.log(value, 2.0) - logZ_ngrams) for gram, value in ngrams.items()}
      
      prefix = Counter(self.ngrams(tokens, n-1))
      logZ_prefix = math.log(sum(prefix.values()), 2.0)
      prefix_probs = {gram: -1*(math.log(value, 2.0) - logZ_prefix) for gram, value in prefix.items()}
      self.ngram_logprobs = ngram_probs
      self.prefix_logprobs = prefix_probs

      return ngram_probs

    def ngrams(self, text, n):
      """Get ngram counts for input"""
      output = []
      for i in range(len(text)-n+1):
          output.append(tuple(text[i:i+n]))
      return output

def test_additive_smoothing_ngram_model():
    test = ["foo . bar","foo foo foo foo foo","--%@-----.+==#@@---","",["foo","bar"]]
    while len(test) > 0:
      text = test.pop()
      try:
        assert type(text) == str, "Text must be a string"
        assert len(text.split()) >= 2, "Text must contain more than one word"
        assert text != "", "Text cannot be empty"
      except AssertionError:
        pass
    text = "foo bar foo bar foo"
    model = AdditiveSmoothingNGramModel(text)
    assert list(model.ngram_logprobs.values()) == [-math.log(2/4,2.0), -math.log(2/4,2.0)]
    text = "In the beginning, was the word, and the word, was with God, and the word was God."
    model = AdditiveSmoothingNGramModel(text)
    assert model.ngram_logprobs[("the", "word")] == -math.log(3/21,2.0)
    text = "foo ./ bar"
    model = AdditiveSmoothingNGramModel(text)
    assert list(model.prefix_logprobs.values()) == [-math.log(1/3,2.0), -math.log(1/3,2.0), -math.log(1/3,2.0)]
    text = "foo .    /   bar"
    model = AdditiveSmoothingNGramModel(text)
    assert list(model.prefix_logprobs.values()) == [-math.log(1/4,2.0), -math.log(1/4,2.0), -math.log(1/4,2.0), -math.log(1/4,2.0)]
    print("Test complete!")
  
def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
    ngrams = model.ngrams(tokens, model.n)
    prefix = model.ngrams(tokens, model.n -1)
    logp_words = model.prefix_logprobs[prefix[0]]
    for i in range(len(tokens)-1):
      try:
          logp_words += model.ngram_logprobs[ngrams[i]] - model.prefix_logprobs[prefix[i]]
      except IndexError:
        pass
    return logp_words

def logp_word_set(model, tokens):
    """Get plog sum of each tokens' permutations"""
    logprobs = []
    for reordered_tokens in itertools.permutations(tokens):
      try:
        logprobs.append(model.ngram_logprobs[tuple(reordered_tokens)])
      except KeyError:
        logprobs.append(0)
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
        tokens = tokens[window_size-(window_size-1):]
    logps_words = np.array([logp_words(model, window) for window in windows])
    logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
    H_words = np.mean(logps_words)
    H_word_sets = np.mean(logps_word_sets)
    return H_words, H_word_sets

if __name__ == '__main__':
    import nose
    nose.runmodule()

