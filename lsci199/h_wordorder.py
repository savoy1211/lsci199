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
    def __init__(self, text, alpha=1, n=2, add_tags=True):
        self.text = text
        self.alpha = alpha
        self.n = n
        self.add_tags = add_tags
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
          current_tokens = (self.n-1)*["<<BOS>>"] + current_tokens + (self.n-1)*["<<EOS>>"]
          tokens += current_tokens
        return tokens
      else:
        return [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]

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
      return ngram_probs, prefix_probs

    def total_prob(self, tokens):
      ngrams = self.ngrams(tokens, self.n)
      prefix = self.ngrams(tokens, self.n-1)
      logp_words = 0
      if len(tokens) == 1:
        try:
          logp_words = self.prefix_logprobs[tuple(tokens)]
        except Exception:
            return n_inf
      elif len(tokens) > 1:
        try:
          logp_words = self.prefix_logprobs[tuple(tokens[:1+(self.n-2)])]
          print("init", tokens[:1+(self.n-2)])
          for i in range(len(tokens)-self.n+1):
            logp_words += self.ngram_logprobs[ngrams[i]] - self.prefix_logprobs[prefix[i]]
            print(ngrams[i], '-', prefix[i])
        except Exception:
          print("EXCEPTION FOUND", tokens, tokens[:1+(self.n-2)])
          print(self.prefix_logprobs, self.ngram_logprobs)
          return n_inf
      return logp_words

    def ngrams(self, tokens, n):
      """Get ngram counts for input"""
      # remove ngram if contains both <<BOS>> and <<EOS>>
      # return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1) if tokens[i:i+n] and '<<BOS>>' not in tuple(tokens[i:i+n]) or '<<EOS>>' not in tuple(tokens[i:i+n])]
      to_remove = []
      all_tuples = []
      for i in range(len(tokens)-n+1):
        p = tuple(tokens[i:i+n])
        all_tuples.append(p)
        for j in range(len(p)-1):
          if ('<<EOS>>','<<BOS>>') == p[j:j+2]:
              to_remove.append(p)
      for i in to_remove:
        try:
          all_tuples.remove(i)
        except ValueError:
          pass
      # print()
      # print("all_tuples", all_tuples)
      # print("to_remove", to_remove)
      # print()
      return all_tuples      




class GPT2Model:
  def __init__(self, model, tokenizer, text):
    self.text = text
    self.window_size = 1
    self.model = model
    self.tokenizer = tokenizer

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
        if self.tokenizer.decode(token) != '<|EOSoftext|>':
            mProb.append((self.tokenizer.decode(token).strip(' '), prob.item()))
    # mProb.append(('<EOS>', 0.0))

    return mProb

def test_bigram():
  # window_size = 1
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=1)
  tags_plog, words_plog = [math.log(2/16,2.0)]*4, [math.log(1/16,2.0)]*12
  h_words_actual = -np.mean(tags_plog + words_plog)
  # print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  

  # window_size = 2
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=2)
  plog_array = [math.log(1/14,2.0)] * 14
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 3
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=3)
  plog_array = [math.log(1/14,2.0)*2 - math.log(1/16,2.0)]*12
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 4
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=4)
  plog_array = [math.log(1/14,2.0)*3-math.log(1/16,2.0)*2]*10
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  
 
def test_bigram2():
  # window_size = 5
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=5)
  plog_array = [math.log(1/14,2.0)*4-math.log(1/16,2.0)*3]*8 + [math.log(1/14,2.0)*3-math.log(1/16,2.0)*2]
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_trigram():
  # window_size = 2 (trigram)
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=2)
  plog_array1, plog_array2 = [math.log(2/18,2.0)] * 4,  [math.log(1/18,2.0)] * 14
  h_words_actual = -np.mean(plog_array1+plog_array2)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 1 (trigram) 
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingNGramModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=1)
  plog_array1, plog_array2 = [math.log(4/20,2.0)]*8, [math.log(1/20,2.0)]*12
  h_words_actual = -np.mean(plog_array1+plog_array2)
  print(h_words_test, h_words_actual)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

def test_4gram():
  model = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  test = logp_words(model, ['my', 'name', 'is', 'ryan'])
  actual = math.log(1/24,2.0) + math.log(1/18,2.0)
  print("test", test)
  print("actual", actual)
  assert test == actual 


def test_ngrams():
  model2 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=2)
  model3 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=3)
  model4 = AdditiveSmoothingNGramModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  assert logp_words(model2, ['ryan']) == math.log(1/16,2.0)
  assert logp_words(model3, ['ryan']) == math.log(1/20,2.0)
  assert logp_words(model4, ['ryan']) == math.log(1/24,2.0)

  b, e = '<<BOS>>', '<<EOS>>'
  a = [b,b,"hello",e,e,b,b,'my','name','is','ryan',e,e,]
  model = AdditiveSmoothingNGramModel("Hello. My name is ryan.", n=3)
  expected = [(b,b,"hello"),(b,"hello",e),("hello",e,e),(b,b,"my"),(b,"my","name"),("my","name","is"),("name","is","ryan"),("is","ryan",e),("ryan",e,e)]
  test = model.ngrams(a,3)
  print("expected", expected)
  print("test",test)
  assert test == expected

  # model = AdditiveSmoothingNGramModel("Hello. Hi, there. My name is Ryan. Ryan Lee. I am a student at University of California, Irvine. I live in Los Angeles", n=4)
  # expected = [(b,b,b),(b,b,"hello"),(b,"hello",e),("hello",e,e),(e,e,e)]
  # test = model.ngrams(model.tokens, 4)
  # print('expected', expected)
  # print('test',test)
  # assert test == expected

def test_windows():
  model = AdditiveSmoothingNGramModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=5)
  assert len(windows) == 16

  model = AdditiveSmoothingNGramModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=7)
  assert len(windows) == 10

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

def H_words(model, set_of_sequences):
    logprobs = np.array([logp_words(model, sequence) for sequence in set_of_sequences])
    return -(np.exp(logprobs) * logprobs).sum(axis=-1)

def H_word_sets(model, set_of_sequences):
    outer_logps = np.array([logp_words(model, sequence) for sequence in set_of_sequences])
    inner_logps = np.array([logp_word_set(model, sequence) for sequence in set_of_sequences])
    return -(np.exp(outer_logps)*inner_logps).sum(axis=-1)

def get_windows(model, window_size):
  # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
  windows = []
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
  sentences = sent_detector.tokenize(model.text.strip(), realign_boundaries=False)
  for sentence in sentences:    
    sentence = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
    sentence = (model.n-1)*["<<BOS>>"] + sentence + (model.n-1)*["<<EOS>>"]
    if len(sentence) == 2*(model.n-1): # if sentence only contains BOS and EOS tags, e.g. ['<<BOS>>', '<<BOS>>', '<<EOS>>', '<<EOS>>']
      break
    window = []
    append_number = 0
    sentence_trunc = sentence
    num_windows = lambda tokens, window_size: 1 if tokens < window_size else tokens-window_size+1
    while append_number != num_windows(len(sentence), window_size):
      window = sentence_trunc[:window_size]
      windows.append(window)
      # if "<<BOS>>" in window and "<<EOS>>" in window:
          # print(window)
          # print(sentence)
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