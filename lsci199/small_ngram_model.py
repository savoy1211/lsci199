import math
from collections import Counter
from modulate_text import *

INF = float('inf')
UNK = "!!!UNK!!!"

def safelog(x):
    try:
        return math.log(x)
    except ValueError:
        return -INF

def left_truncate(xs, n):
    """ Return a tuple with the last n elements of xs """
    assert n >= 0
    if n == 0:
        return ()
    else:
        return tuple(xs[-n:])

def test_left_truncate():
    assert left_truncate("abcdef", 2) == tuple("ef")
    assert left_truncate([1,2,'a'], 1) == ('a',)
    assert left_truncate("abc", 0) == ()
    assert left_truncate("ab", 10) == tuple("ab")    

def right_truncate(xs, n):
    assert n >= 0
    """ Return a tuple with the first n elements of xs """
    return tuple(xs[:n])

def ngrams(xs, n):
    """ Return n-grams of xs """
    assert n >= 0
    return zip(*[xs[i:] for i in range(n)])

def test_ngrams():
    assert list(ngrams("abc", 2)) == [('a', 'b'), ('b', 'c')]
    assert list(ngrams("a", 2)) == []

def tokens_in_context(xs):
    """ Return all the elements of xs and their context (preceding elements) """
    context = []
    for x in xs:
        yield x, tuple(context)
        context.append(x)

def test_tokens_in_context():
    assert list(tokens_in_context("abd")) == [('a', ()), ('b', ('a',)), ('d', ('a', 'b'))]
    assert list(tokens_in_context("")) == []

class NgramModel:
    def __init__(self, tokens, n=3, alpha=1, include_unk_in_vocab=True):
        assert n > 0
        assert alpha >= 0
        self.n = n
        self.alpha = alpha
        
        self.vocab = set()
        for token in tokens:
            self.vocab.add(token)
        if include_unk_in_vocab:
            self.vocab.add(UNK)

        self.freq = Counter()
        for ngram in ngrams(tokens, n):
            self.freq[ngram] += 1
            for i in range(n):
                sub_ngram = right_truncate(ngram, i)
                self.freq[sub_ngram] += 1

    def conditional_logprob(self, token, context):
        """ Calculate logp(token | context) """
        combined = context + (token,)
        numerator = self.freq[combined] + self.alpha
        denominator = self.freq[context] + len(self.vocab)*self.alpha
        if denominator == 0:
            return -INF
        else:
            assert denominator >= numerator
            return safelog(numerator) - math.log(denominator)
        
    def window_logprob(self, tokens):
        result = 0.0
        for token, context in tokens_in_context(tokens):
            context = left_truncate(context, self.n - 1)
            result += self.conditional_logprob(token, context)
        return result

def test_mle_ngram_model():
    model = NgramModel("one two one three one two".split(), n=2, alpha=0, include_unk_in_vocab=False)
    assert round(math.exp(model.conditional_logprob('two', ('one',))), 5) == round(2/3, 5)
    assert round(math.exp(model.conditional_logprob('three', ('one',))), 5) == round(1/3, 5)
    # p(one two) = p(one)p(two|one) = 3/5 * 2/3 = 2/5
    assert round(math.exp(model.window_logprob("one two".split())), 5) == round(2/5, 5)
    # assert total probability for all possible windows is 1
    assert sum(math.exp(model.window_logprob([w1, w2])) for w1 in model.vocab for w2 in model.vocab) == 1
    
    model = NgramModel("this is a test a test".split(), n=2, alpha=0, include_unk_in_vocab=False)
    # p(a test) = p(a)p(test|a) = 2/5 * 1 = 2/5
    assert round(math.exp(model.window_logprob("a test".split())), 5) == round(2/5, 5)
    # p(test a) = p(test)p(a | test) = 1/5 * 1 = 1/5
    assert round(math.exp(model.window_logprob("test a".split())), 5) == round(1/5, 5)
    assert sum(math.exp(model.window_logprob([w1, w2])) for w1 in model.vocab for w2 in model.vocab) == 1    

    model = NgramModel("this is a test of a test of a test".split(), n=3, alpha=0, include_unk_in_vocab=False)
    # p(a test of) = p(a) p(test|a) p(of | a test) = 2/8 * 1 * 1 = 1/4
    assert round(math.exp(model.window_logprob("a test of".split())), 5) == round(1/4, 5)
    assert round(sum(math.exp(model.window_logprob([w1, w2, w3])) for w1 in model.vocab for w2 in model.vocab for w3 in model.vocab), 5) == 1

def test_additive_smoothing_ngram_model():
    model = NgramModel("one two one two one three".split(), n=2, alpha=.5, include_unk_in_vocab=False)
    # 2 + .5 / (1 + .5 + 2 + .5 + 0 + .5) = 5/9
    assert round(math.exp(model.conditional_logprob('two', ('one',))), 5) == round(5/9, 5)
    # 1 + .5 / (1 + .5 + 2 + .5 + 0 + .5) = 1/3    
    assert round(math.exp(model.conditional_logprob('three', ('one',))), 5) == round(1/3, 5)
    assert round(sum(math.exp(model.window_logprob([w1, w2])) for w1 in model.vocab for w2 in model.vocab), 5) == 1

    model = NgramModel("one two one two one three".split(), n=3, alpha=.5, include_unk_in_vocab=False)
    # 2 + .5 / (2 + .5 + .5 + .5) = 5/7
    assert round(math.exp(model.conditional_logprob('one', ('one', 'two',))), 5) == round(5/7, 5)
    assert round(sum(math.exp(model.window_logprob([w1, w2, w3])) for w1 in model.vocab for w2 in model.vocab for w3 in model.vocab), 5) == 1    

if __name__ == '__main__':
    import nose
    nose.runmodule()
