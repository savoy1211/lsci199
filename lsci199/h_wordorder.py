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
# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
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

class AdditiveSmoothingModel:
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
    # print()
    # print("tokens: ", tokens)
    # print("ngrams: ",ngrams)
    # print("prefix: ",prefix)
    logp_words = 0
    try:
      logp_words += self.prefix_logprobs[tuple(tokens[:1])]
      if self.n > len(tokens): # only use prefix_logprobs
        for i in range(1,len(tokens)):
          logp_words += self.prefix_logprobs[tuple(tokens[:i+1])] - self.prefix_logprobs[tuple(tokens[:i])]
          print(tokens[:i+1], '-', tokens[:i])
      elif self.n <= len(tokens):
        for i in range(1,self.n-1):
          logp_words += self.prefix_logprobs[tuple(tokens[:i+1])] - self.prefix_logprobs[tuple(tokens[:i])] 
          print(tokens[:i+1],"-",tokens[:i])
        for i in range(len(tokens) - self.n+1):
          logp_words += self.ngram_logprobs[tuple(tokens[i:i+self.n])] - self.prefix_logprobs[tuple(tokens[i:i+self.n-1])]
          print(tokens[i:i+self.n],'-',tokens[i:i+self.n-1])
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
    return all_tuples      

def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
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
  H_words = -np.mean(logps_words)
  H_word_sets = -np.mean(logps_word_sets)
  return H_words, H_word_sets
