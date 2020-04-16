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
import time

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

def make_embedding_index(tokens):
  index = {}
  max_index = 0
  for token in tokens:
    if token not in index:
      index[token] = max_index
      max_index += 1
  index["<!!!!UNK!!!!>"] = max_index
  return index

def embed(tokens, index):
  a = np.zeros(len(index))
  for token in tokens:
    if token in index:
      a[index[token]] += 1
    else:
      a[index["<!!!!UNK!!!!>"]] += 1
  return a

def make_embedding(tokens):
  index = make_embedding_index(tokens)
  return embed(tokens, index), index

def preprocess(text):
  tokens = nltk.tokenize.word_tokenize(text)
  utokens = [token.casefold() for token in tokens]
  return utokens

def ngrams(input, n):
  """Get ngrams for input"""
  output = []
  for i in range(len(input)-n+1):
      output.append(tuple(input[i:i+n]))
  return output

def collect_ngram_counts(text, n):
    return Counter(ngrams(preprocess(text),n))

def reverse_index(index):
  """ reverse_index(index) is a list xs where xs[index[w]] == w """
  return [word for _, word in sorted((v, k) for k, v in index.items())]

def plog_words(text_vector, index, alpha=1): # this is calculating the entropy so the name should reflect that
    """Calculate the entropy of a corpus's words"""
    tot_prob = 0
    for word in index:
        single_prob = (text_vector[index[word]]+alpha) / (text_vector.sum()) # RF: Need to include alpha in the denominator too. Also, isn't this just the unigram frequency?
        tot_prob += (math.log(single_prob, 2.0)) * single_prob
    return -1*tot_prob 

def get_ngram_plogs(text, n):
    """Calculate plog of ngrams and prefix"""
    ngrams = collect_ngram_counts(text, n)
    logZ_ngrams = math.log(sum(ngrams.values()), 2.0)
    ngram_probs = {gram: -1*(math.log(value, 2.0) - logZ_ngrams) for gram, value in ngrams.items()}
    
    prefix = collect_ngram_counts(text, n - 1)
    logZ_prefix = math.log(sum(prefix.values()), 2.0)
    prefix_probs = {gram: math.log(value, 2.0) - logZ_prefix for gram, value in prefix.items()}
    return ngram_probs, prefix_probs

def ngram_model(tokens, ngram_plogs, prefix_plogs, n):
    """Calculate plog(w1, w2, ... wn | w1, w2, ... wn-1)"""
    INF = float('inf')
    tokens = tuple(tokens)

    # plog of n-1
    starting_prefix = tokens[:(n-1)]
    if starting_prefix in prefix_plogs:
        logp = prefix_plogs[starting_prefix]
    else:
        logp = -INF

    # plog of n
    for i in range(n, len(tokens)+n-1):
        try:
            ngram = ngram_plogs[tokens[i-n:i]]
            prefix = prefix_plogs[tokens[i-n:i-1]]
        except KeyError:
            return 0
        logp += ngram - prefix
    return logp
    
def plog_wordset(text,bigram_plogs,unigram_plogs, n):
    """Calculate the entropy of a text's set of words given their order"""
    alpha = 1
    permutations = list(itertools.permutations(text.split(" ")))
    num_plog = ngram_model(preprocess(text),bigram_plogs,unigram_plogs,n) # The real probability
    denom_plog = 0

    for each in permutations:
        each = ' '.join(str(x) for x in each)
        denom_plog += ngram_model(preprocess(each),bigram_plogs,unigram_plogs,n)  # Watch out! These are log probabilities, so you shouldn't add them here
    return num_plog / (denom_plog + alpha) # This is p(real order | set of words) (except for the issue with adding logps)

def get_h_words(text):
  """Get avreage plog of words"""
  big_index = make_embedding_index(preprocess(text))
  reversed_index = reverse_index(big_index)
  df = pd.DataFrame({
      'word': reversed_index,
      text: embed(preprocess(text), big_index),
  })
  return plog_words(df[text], big_index, alpha=1)

def get_h_wordset(text,n,p):
  """Get average plog of each sentence[:p] of text"""
  bigram_plogs, unigram_plogs = get_ngram_plogs(text, n)
  p_wordset = ngram_model(preprocess(text), bigram_plogs, unigram_plogs, n)
  p_order = 0

  # Tokenize text by sentence
  sent_detector = nltk.data.load('tokenizers/punkt/english.pickle') # Nice!
  b_sents = sent_detector.tokenize(text.strip(), realign_boundaries=False)

  # Iterate through each sentence[:p] of text
  for sent in b_sents:
    tokens = preprocess(sent)
    trunc_text = " ".join(tokens[:p]) 
    p_order += plog_wordset(trunc_text,bigram_plogs,unigram_plogs, n)

  return p_order/len(b_sents) # this is the average logp(order | set of words)

def get_h_wordorder(text, ngram_n=2, p_window=3):
  h_words = get_h_words(text) # It looks like this just gets the unigram probability
  h_wordset = get_h_wordset(text, ngram_n, p_window) # this is really conditional entropy of words
  h_wordorder = h_words - h_wordset
  print("H(words): " +str(h_words))
  print("H(word set): "+str(h_wordset)) 
  print("H(word order | word set): " + str(h_wordorder))
  return h_wordorder

# Let's refactor to something like the below.
# We specify "models" which have a method "logprob". An n-gram model is one such model.
# Later, we can drop in neural network models using the same interface.

class AdditiveSmoothingNGramModel:
    def __init__(self, text, alpha=1):
        # ... do stuff to extract the n-gram counts from the text ...
        # I think you can do this relatively simply, without all the infrastructure above
        self.alpha = alpha
        ...

    def logprob(self, tokens):
        # ... return the n-gram probaiblity for the text according to the counts collected in __init__ ...
        ...


def test_additive_smoothing_ngram_model():
    # ... write some tests: simple cases where you know the right answer ...
    model = AdditiveSmoothingNGramModel(text)
    assert model.logprob(something) == something
    
def logp_words(model, tokens):
    return model.logprob(tokens)

def logp_word_set(model, tokens):
    logprobs = [model.logprob(model, reordered_tokens) for reordered_tokens in itertools.permutations(tokens)]
    return scipy.special.logsumexp(logprobs)

def H_words(model, set_of_sequences):
    logprobs = np.array([logp_words(sequence) for sequence in set_of_sequences])
    return -(np.exp(logprobs) * logprobs).sum(axis=-1)

def H_word_sets(model, set_of_sequences):
    outer_logps = np.array([logp_words(sequence) for sequence in set_of_sequences])
    inner_logps = np.array([logp_word_set(sequence) for sequence in set_of_sequences])
    return -(np.exp(outer_logps)*inner_logps).sum(axis=-1)

def survey_text(model, tokens, window_size):
    # ... for a sliding window of contiguous words of size window_size, get H[words] and H[word set] ...
    windows = ...
    logps_words = np.array([logp_words(model, window) for window in windows])
    logps_word_sets = np.array([logp_word_set(model, window) for window in windows])
    H_words = -np.mean(logps_words)
    H_word_sets = -np.mean(logps_word_sets)
    return H_words, H_word_sets

        
