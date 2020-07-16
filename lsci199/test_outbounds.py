from ngram_model import *

def test_2grams():
    text = "Hello there. My name is Ryan."
    model = NGramModel(text, alpha=0, n=2, sentence_inbound=False)
    h_words_test, h_wordset_test = survey_text(model, 2)
    log_array = [math.log(1/5,2.0)] * 5
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 
    
    text = "Hello there. My name is Ryan."
    model = NGramModel(text, alpha=0, n=2, sentence_inbound=True)
    h_words_test, h_wordset_test = survey_text(model, 2)
    log_array = [math.log(1/5,2.0)] * 4
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_3grams():
    text = "Hello there. My name is Ryan."
    model = NGramModel(text, alpha=0, n=3, sentence_inbound=False)
    h_words_test, h_wordset_test = survey_text(model, 3)
    log_array = [math.log(1/4,2.0)] * 4
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 
    
    text = "Hello there. My name is Ryan."
    model = NGramModel(text, alpha=0, n=3, sentence_inbound=False)
    h_words_test, h_wordset_test = survey_text(model, 3)
    log_array = [math.log(1/4,2.0)] * 2
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 
    
    text = "Hello there. My name is Ryan."
    model = NGramModel(text, alpha=0, n=3, sentence_inbound=True, include_smaller_windows=True)
    h_words_test, h_wordset_test = survey_text(model, 3)
    log_array = [math.log(1/4,2.0)] * 2 + [math.log(1/5,2.0)]
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_windows():
    text = "hello there. my name is Edgar. Hi."
    m = NGramModel(text, alpha=0, n=3, include_smaller_windows=True)
    assert get_windows(m, 3) == [['hello','there'],['my','name','is'],['name','is','edgar'],['hi']]
    
    text = "hello there. my name is Edgar. Hi."
    m = NGramModel(text, alpha=0, n=3)
    assert get_windows(m, 3) == [['my','name','is'],['name','is','edgar']]

if __name__ == '__main__':
    import nose
    nose.runmodule()