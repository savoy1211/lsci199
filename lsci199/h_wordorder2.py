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
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import csv
from random import shuffle

n_inf = -float("inf")

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

# nltk.download("punkt") # This gives us access to nltk.tokenize.word_tokenize
url_prefix = "http://www.socsci.uci.edu/~rfutrell/teaching/lsci109-w2020/data/"
pride_and_prejudice = download_gutenberg_text(url_prefix + "1342-0.txt")
two_cities = download_gutenberg_text(url_prefix + "98-0.txt")
moby_dick = download_gutenberg_text(url_prefix + "2701-0.txt")
hard_times = download_gutenberg_text(url_prefix + "786-0.txt")

class AdditiveSmoothingNGramModel:
    def __init__(self, text, alpha=1, n=2, add_tags=True, randomize_text=False):
        self.text = text
        self.alpha = alpha
        self.n = n
        self.add_tags = add_tags
        self.randomize_text = randomize_text
        self.tokens = self.tokens_init(text, self.n)
        self.tokens_tagless = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
        self.ngram_logprobs, self.prefix_logprobs = self.logprob_init(self.tokens)

    def tokens_init(self, text,n):
      if self.add_tags is True:
        tokens = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text.strip(), realign_boundaries=False)
        for sentence in sentences:
          current_tokens = [token.casefold() for token in nltk.word_tokenize(sentence) if token.isalnum()]
          # current_tokens = (self.n-1)*["<<BOS>>"] + current_tokens + (self.n-1)*["<<EOS>>"]
          tokens += current_tokens
        if self.randomize_text is True:
          shuffle(tokens)
        return tokens
      else:
        tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
        if self.randomize_text is True:
          shuffle(tokens)
        return tokens

    def logprob_init(self, tokens):
      # p(w1, w2, ..., wn) = product from t=1 to t=n of p(w_t | w_{t - (n-1)} ..., w_{t - 1})
      ngrams = Counter(self.ngrams(tokens,self.n))
      total_val = sum(ngrams.values())
      ngram_probs = {gram: float(math.log((value/total_val),2.0)) for gram, value in ngrams.items()}
      prefix_probs = {}
      prefix_num = 1
      while self.n-prefix_num > 0:
        current_n = self.n - prefix_num
        prefix = Counter(self.ngrams(self.tokens_init(self.text, current_n+1), current_n))
        total_val = sum(prefix.values())
        prefix_probs.update({gram: float(math.log((value/total_val),2.0)) for gram, value in prefix.items()})
        prefix_num += 1
      # print()
      # print("ngrams", ngram_probs)
      # print("prefix", prefix_probs)
      return ngram_probs, prefix_probs

    def total_prob(self, tokens):
      ngrams = self.ngrams(tokens, self.n)
      prefix = self.ngrams(tokens, self.n-1)
      logp_words = 0
      try:
        logp_words += self.prefix_logprobs[tuple(tokens[:1])]
        if self.n > len(tokens): # only use prefix_logprobs
          for i in range(1,len(tokens)):
            logp_words += self.prefix_logprobs[tuple(tokens[:i+1])] - self.prefix_logprobs[tuple(tokens[:i])] 
            # print(tokens[:i+1], '-', tokens[:i])
        elif self.n <= len(tokens):
          for i in range(1,self.n-1):
            logp_words += self.prefix_logprobs[tuple(tokens[:i+1])] - self.prefix_logprobs[tuple(tokens[:i])] 
            # print(tokens[:i+1],"-",tokens[:i])
          for i in range(len(tokens) - self.n+1):
            logp_words += self.ngram_logprobs[tuple(tokens[i:i+self.n])] - self.prefix_logprobs[tuple(tokens[i:i+self.n-1])]
            # print(tokens[i:i+self.n],'-',tokens[i:i+self.n-1])
      except Exception:
        return n_inf
      return logp_words

    def ngrams(self, tokens, n):
      """Get ngram counts for input"""
      to_remove = []
      all_tuples = []
      for i in range(len(tokens)-n+1):
        p = tuple(tokens[i:i+n])
        all_tuples.append(p)
        # for j in range(len(p)-1):
        #   if ('<<EOS>>','<<BOS>>') == p[j:j+2]:
        #       to_remove.append(p)
      # for i in to_remove:
      #   try:
      #     all_tuples.remove(i)
        # except ValueError:
        #   pass
      return all_tuples      

def test_bigram():
  # window_size = 1
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=1)
  words_plog = [math.log(1/12,2.0)]*12
  h_words_actual = -np.mean(words_plog)
  # assert round(h_words_test, 12) == round(h_words_actual, 12) 
  assert h_words_test == h_words_actual 

  # window_size = 2
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=2)
  plog_array = [math.log(1/11,2.0)] * 10
  h_words_actual = -np.mean(plog_array)
  # assert round(h_words_test, 12) == round(h_words_actual, 12)
  assert h_words_test == h_words_actual 

  # window_size = 3
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=3)
  plog_array = [math.log(1/12,2.0) + math.log(1/11,2.0) - math.log(1/12,2.0) + math.log(1/11,2.0) -math.log(1/12,2.0)] * 8 
  h_words_actual = -np.mean(plog_array)
  print(h_words_test, h_words_actual)
  # assert round(h_words_test, 12) == round(h_words_actual, 12)
  assert h_words_test == h_words_actual 

  # window_size = 4
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=4)
  plog_array = [math.log(1/12,2.0) + 3*(math.log(1/11,2.0) - math.log(1/12,2.0))]*7
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  
 
  # window_size = 5
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=5)
  plog_array = [math.log(1/12,2.0) + 4*(math.log(1/11,2.0) - math.log(1/12,2.0))]*6
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_trigram():
  # window_size = 3 (trigram)
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=3)
  plog_array1 = [math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)+math.log(1/10,2.0)-math.log(1/11,2.0)] * 8
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 2 (trigram) 
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=2)
  plog_array1 = [math.log(1/11,2.0)] + [math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)]*9
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  model = AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3)
  test = model.total_prob(["my","name","is","ryan"])
  actual = math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)+2*(math.log(1/10,2.0)-math.log(1/11,2.0))
  assert test == actual

def test_4gram():
  model = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  test = model.total_prob(['my', 'name', 'is', 'ryan'])
  actual = math.log(1/9,2.0)
  assert test == actual 

  test = model.total_prob(['my','name','is','ryan','and'])
  actual = 2*math.log(1/9,2.0) - math.log(1/10,2.0)
  assert test == actual 

  test = model.total_prob(['my','name'])
  actual = math.log(1/11,2.0)
  assert test == actual

  w = get_windows(model, 4)
  assert w == [['my', 'name', 'is', 'ryan'],['name', 'is', 'ryan', 'and'],['is', 'ryan', 'and', 'i'],['ryan', 'and', 'i', 'am'],['and', 'i', 'am', 'from'],['i', 'am', 'from', 'los'],['am', 'from', 'los', 'angeles']]

def test_ngrams():
  model2 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=2)
  model3 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=3)
  model4 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  assert logp_words(model2, ['ryan']) == logp_words(model3, ['ryan']) == logp_words(model4, ['ryan'])
  # assert logp_words(model3, ['ryan']) == math.log(1/20,2.0)
  # assert logp_words(model4, ['ryan']) == math.log(1/24,2.0)

  # b, e = '<<BOS>>', '<<EOS>>'
  # a = [b,b,"hello",e,e,b,b,'my','name','is','ryan',e,e,]
  # model = AdditiveSmoothingNGramModel("Hello. My name is ryan.", n=3)
  # expected = [(b,b,"hello"),(b,"hello",e),("hello",e,e),(b,b,"my"),(b,"my","name"),("my","name","is"),("name","is","ryan"),("is","ryan",e),("ryan",e,e)]
  # test = model.ngrams(a,3)
  # print("expected", expected)
  # print("test",test)
  # assert test == expected

  # model = AdditiveSmoothingNGramModel("Hello. Hi, there. My name is Ryan. Ryan Lee. I am a student at University of California, Irvine. I live in Los Angeles", n=4)
  # expected = [(b,b,b),(b,b,"hello"),(b,"hello",e),("hello",e,e),(e,e,e)]
  # test = model.ngrams(model.tokens, 4)
  # print('expected', expected)
  # print('test',test)
  # assert test == expected

def test_windows():
  model = AdditiveSmoothingNGramModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=5)
  assert len(windows) == 9

  model = AdditiveSmoothingNGramModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=7)
  assert len(windows) == 5

def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
    result = model.total_prob(tokens)
    return result

def logp_word_set(model, tokens): 
    """Get plog sum of each tokens' permutations"""
    logprobs = [logp_words(model, reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
    if not logprobs:
        logprobs = [0]
    return scipy.special.logsumexp(logprobs)

def get_windows(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = []
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(model.text.strip(), realign_boundaries=False)
  for sentence in sentences:    
    sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
    window = []
    append_number = 0
    sentence_trunc = sentence
    num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
    while append_number != num_windows(len(sentence), window_size):
      window = sentence_trunc[:window_size]
      if len(window) > 0 and len(window) == window_size:
        windows.append(window)
      append_number += 1
      sentence_trunc = sentence_trunc[1:]
  return windows

def get_windows_full(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = []
  t = [token.casefold() for token in nltk.tokenize.word_tokenize(model.text) if token.isalnum()]
  window = []
  append_number = 0
  sentence_trunc = t
  num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
  while append_number != num_windows(len(t), window_size):
    window = sentence_trunc[:window_size]
    if len(window) > 0 and len(window) == window_size:
      windows.append(window)
    append_number += 1
    sentence_trunc = sentence_trunc[1:]
  return windows

def survey_text(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = get_windows(model,window_size)
  logps_words = np.array([logp_words(model, window) for window in windows])
  logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
  H_words = -np.mean(logps_words)
  H_word_sets = -np.mean(logps_word_sets)
  return H_words, H_word_sets

if __name__ == '__main__':
    import nose
    nose.runmodule()