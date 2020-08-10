import itertools

import nltk

def file_slider(filename, n):
    with open(filename, 'rt') as infile:
        for line in infile:
            tokenized = nltk.tokenize.word_tokenize(line)
            windows = sliding(tokenized, n)
            for window in windows:
                yield window

def sliding(iterable, n):
    assert n >= 0
    its = itertools.tee(iterable, n)
    for i, iterator in enumerate(its):
        for _ in range(i):
            try:
                next(iterator)
            except StopIteration:
                return zip([])
    return zip(*its)

def test_sliding():
    assert list(sliding("abc", 2)) == [('a', 'b'), ('b', 'c')]
    assert list(sliding("abc", 3)) == [('a', 'b', 'c')]
    assert list(sliding("abc", 4)) == []
    assert list(sliding("abc", 0)) == []
    assert list(sliding("a", 1)) == [('a',)]
    assert list(sliding("", 4)) == []


if __name__ == "__main__":
    import nose
    nose.runmodule()
