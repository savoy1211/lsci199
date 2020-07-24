from download_local_gutenberg_texts import *
from ngram_model import *

# text = mega_text_10
text = 'Hello there. Hi. My name is Ryan and I am from Los Angeles.'

def test_ordered_inbound():
    # Train ordered, test ordered (sentence inbound)
    m = NGramModel(text, alpha=0.1, n=3)
    test_windows = get_windows(m, 3)
    actual_windows = [['my','name','is'],['name','is','ryan'],['is','ryan','and'],['ryan','and','i'],['and','i','am'],['i','am','from'],['am','from','los'],['from','los','angeles']]
    print(test_windows)
    assert test_windows == actual_windows
    
def test_random_inbound():
    # Train random, test random (sentence inbound)
    m = NGramModel(text, alpha=0.1, n=3, randomize_text=True, randomize_sentence_inbound=True, ordered_windows=False)
    random_text = m.text_randomized
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 8
    print(test_windows_len)
    assert test_windows_len == actual_windows_len
    
def test_ordered_outbound():
    # Train ordered, test ordered (sentence outbound)
    m = NGramModel(text, alpha=0.1, n=3, sentence_inbound=False)
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 11
    print(test_windows_len)
    assert test_windows_len == actual_windows_len
    
def test_random_outbound():
    # Train random, test random (sentence outbound)
    m = NGramModel(text, alpha=0.1, n=3, randomize_text=True, randomize_sentence_inbound=False, sentence_inbound=False)
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 11
    print(test_windows_len)
    assert test_windows_len == actual_windows_len

text = "Trump has presided over the complete dismantling of American influence in the world and the destruction of our economy. I know the stock market has looked good, but the stock market has become totally uncoupled from the economy. According to the stock market, the future is just as bright now as it was in January of this year, before most of us had even heard of a novel coronavirus. That doesn’t make a lot of sense. And a lot can happen in the next few months. The last two weeks feel like a decade. And my concern is that if Trump now gets to be the law-and-order President, that may be his path to re-election, if such a path exists. Of course, this crisis has revealed, yet again, how unfit he is to be President. The man couldn’t strike a credible note of reconciliation if the fate of the country depended on it—and the fate of the country has depended on it. I also think it’s possible that these protests wouldn’t be happening, but for the fact that Trump is President. Whether or not the problem of racism has gotten worse in our society, having Trump as President surely makes it seem like it has. It has been such a repudiation of the Obama presidency that, for many people, it has made it seem that white supremacy is now ascendant. So, all the more reason to get rid of Trump in November."

def test_random():
    m = NGramModel(text, alpha=0.1, n=3, randomize_text=True, sentence_inbound=True, randomize_sentence_inbound=True, ordered_windows=False)
    # t = TestCorpus(text[:int(len(text)/10)], randomize_text=True, sentence_inbound=True, ordered_windows=False)
    t = TestCorpus(text[:int(len(text)/10)], randomize_text=True, sentence_inbound=True, randomize_sentence_inbound=True, ordered_windows=False)
    print(m.text)
    print(t.text)
    
if __name__ == '__main__':
    import nose
    nose.runmodule()