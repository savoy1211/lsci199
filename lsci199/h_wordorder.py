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
from functools import lru_cache


n_inf = -float("inf")


class AdditiveSmoothingNGramModel:
    def __init__(self, text, alpha=1, n=2, add_tags=True):
        self.text = text
        self.alpha = alpha
        self.n = n
        self.add_tags = add_tags
        self.tokens = self.tokens_init(text)
        # self.tokens_tagless = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
        self.ngram_logprobs, self.prefix_logprobs = self.logprob_init(self.tokens)

    def tokens_init(self, text):
      if self.add_tags is True:
        tokens = []
        sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = sent_detector.tokenize(text.strip(), realign_boundaries=False)
        for sentence in sentences:
          current_tokens = [token.casefold() for token in nltk.word_tokenize(sentence) if token.isalnum()]
          current_tokens = (self.n-1)*["<<START>>"] + current_tokens + (self.n-1)*["<<END>>"]
          tokens += current_tokens
        return tokens
      else:
        return [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]

    def logprob_init(self, tokens):
      # p(w1, w2, ..., wn) = product from t=1 to t=n of p(w_t | w_{t - (n-1)} ..., w_{t - 1})
      ngrams = Counter(self.ngrams(tokens,self.n))
      total_val = sum(ngrams.values())
      ngram_probs = {gram: float(math.log((value/total_val),2.0)) for gram, value in ngrams.items()}

      prefix = Counter(self.ngrams(tokens, self.n-1))
      total_val = sum(prefix.values())
      prefix_probs = {gram: float(math.log((value/total_val),2.0)) for gram, value in prefix.items()}

      return ngram_probs, prefix_probs

    def total_prob(self, tokens):
      ngrams = self.ngrams(tokens, self.n)
      prefix = self.ngrams(tokens, self.n-1)
      logp_words = 0
      if len(tokens) == 1:
        try:
          logp_words = self.prefix_logprobs[prefix[0]]
        except Exception:
            return n_inf
      elif len(tokens) > 1:
        logp_words = self.prefix_logprobs[prefix[0]]
        for i in range(len(tokens)-1):
            try:
              logp_words += self.ngram_logprobs[ngrams[i]] - self.prefix_logprobs[prefix[i]]
            except Exception: 
              return n_inf
      return logp_words

    def ngrams(self, text, n):
      """Get ngram counts for input"""
      return [tuple(text[i:i+n]) for i in range(len(text)-n+1) if text[i:i+n]]

class GPT2Model:
  def __init__(self, model, tokenizer, text):
    self.text = text
    self.window_size = 1
    self.model = model
    self.tokenizer = tokenizer

  @profile
  def total_prob(self, sentence, with_delimiters=True):
    if with_delimiters:
        sentence_tokens = [self.tokenizer.bos_token_id] + self.tokenizer.encode(sentence) + [self.tokenizer.eos_token_id]
    else:
        sentence_tokens = self.tokenizer.encode(sentence)
    sentence_tensor = torch.tensor([sentence_tokens])
    if torch.cuda.is_available():
        sentence_tensor = sentence_tensor.to('cuda')
        self.model.to('cuda')
    with torch.no_grad():
        predictions = self.model(sentence_tensor)
        probabilities = torch.log_softmax(predictions[0], -1)
        # print(probabilities[0].shape)
        model_probs = probabilities[0, :, tuple(sentence_tokens)].diag(int(with_delimiters))
        entropy = - (probabilities[0,:,:].exp() * probabilities[0,:,:]).sum(-1)[1:]

    # entropy_reduction = []
    # for idx, item in enumerate(entropy):
    #     if idx < len(entropy) - 1:
    #         entropy_reduction.append(entropy[idx].item() - entropy[idx + 1].item())

    mProb = []
    # mProb.append(('<BOS>', 0.0))
    for token, prob in zip(sentence_tokens[1:], model_probs):
        if self.tokenizer.decode(token) != '<|endoftext|>':
            mProb.append((self.tokenizer.decode(token).strip(' '), prob.item()))
    # mProb.append(('<EOS>', 0.0))

    return mProb

def test_wordorder():
  # window_size = 1
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, friend. My name is Ryan and I am from Los Angeles."), window_size=1)
  tags_plog, words_plog = [math.log(2/16,2.0)]*4, [math.log(1/16,2.0)]*12
  h_words_actual = -np.mean(tags_plog + words_plog)
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  

  # window_size = 2
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, friend. My name is Ryan and I am from Los Angeles."), window_size=2)
  plog_array = [math.log(1/15,2.0)] * 15
  h_words_actual = -np.mean(plog_array)
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 3
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, friend. My name is Ryan and I am from Los Angeles."), window_size=3)
  plog_array = [2*math.log(1/15,2.0)-math.log(1/16,2.0)]*12
  h_words_actual = -np.mean(plog_array)
  print(len(plog_array))
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 4
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, friend. My name is Ryan and I am from Los Angeles."), window_size=4)
  plog_array = [3*math.log(1/15,2.0)-2*math.log(1/16,2.0)]*10
  h_words_actual = -np.mean(plog_array)
  print(len(plog_array))
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  

  # window_size = 2 (trigram)
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, friend. My name is Ryan and I am from Los Angeles.", n=3), window_size=2)
  plog_array1, plog_array2 = [math.log(2/18,2.0)] * 4,  [math.log(1/18,2.0)] * 14
  h_words_actual = -np.mean(plog_array1+plog_array2)
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

def logp_words(model, tokens):
    """Get conditional plog of words using ngram model"""
    result = model.total_prob(tokens)
    # print(result)
    entropy_array = [each[1] for each in result]
    # entropy_array = [each[1] for each in result if (len(result) == model.window_size)]
    return sum(entropy_array)

def logp_word_set(model, tokens): 
    """Get plog sum of each tokens' permutations"""
    logprobs = [logp_words(model, reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
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

def survey_text(model, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(model.text.strip(), realign_boundaries=False)
    for sentence in sentences:    
      sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
      sentence = (model.n-1)*["<<START>>"] + sentence + (model.n-1)*["<<END>>"]
      for token in sentence:
        window = []
        if len(sentence[:window_size]) == window_size: 
          window = sentence[:window_size]
          windows.append(window)
          sentence = sentence[1:]

    logps_words = np.array([logp_words(model, window) for window in windows])
    logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
    H_words = -np.mean(logps_words)
    H_word_sets = -np.mean(logps_word_sets)
    return H_words, H_word_sets

def survey_text_gpt2(model, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = []
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(model.text.strip(), realign_boundaries=False)
    for sentence in sentences:    
      sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
      for token in sentence:
        window = []
        if len(sentence[:window_size]) == window_size: 
          window = sentence[:window_size]
          windows.append(window)
          sentence = sentence[1:]
    model.window_size = window_size
    logps_words = np.array([logp_words(model, window) for window in windows])
    logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
    H_words = -np.mean(logps_words)
    H_word_sets = -np.mean(logps_word_sets)
    return H_words, H_word_sets

if __name__ == '__main__':
    import nose
    nose.runmodule()