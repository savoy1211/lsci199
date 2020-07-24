from download_gutenberg_texts import *
from ngram_model import *

# text = mega_text_10
text = 'Hello there. Hi. My name is Ryan and I am from Los Angeles.'

def test_ordered_inbound():
    # Train ordered, test ordered (sentence inbound)
    m = NGramModel(text, alpha=0.1, n=3)
    test_windows = get_windows(m, 3)
    actual_windows = [['my','name','is'],['name','is','ryan'],['is','ryan','and'],['ryan','and','i'],['and','i','am'],['i','am','from'],['am','from','los'],['from','los','angeles']]
    assert test_windows == actual_windows
    
def test_random_inbound():
    # Train random, test random (sentence inbound)
    m = NGramModel(text, alpha=0.1, n=3, randomize_text=True, randomize_sentence_inbound=True, ordered_windows=False)
    random_text = m.text_randomized
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 8
    assert test_windows_len == actual_windows_len
    
def test_ordered_outbound():
    # Train ordered, test ordered (sentence outbound)
    m = NGramModel(text, alpha=0.1, n=3, sentence_inbound=False)
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 11
    assert test_windows_len == actual_windows_len
    
def test_random_outbound():
    # Train random, test random (sentence outbound)
    m = NGramModel(text, alpha=0.1, n=3, randomize_text=True, randomize_sentence_inbound=False, sentence_inbound=False)
    test_windows_len = len(get_windows(m,3))
    actual_windows_len = 11
    assert test_windows_len == actual_windows_len
    
if __name__ == '__main__':
    import nose
    nose.runmodule()