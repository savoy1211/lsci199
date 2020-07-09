from ngram_model import *

def test_2grams():
    text = "a b c d e f g h i"

    # window_size = 1, is_logprob = True
    model = NGramModel(text, alpha=1, n=2)
    h_words_test, h_wordset_test = survey_text(model, 1)
    log_array = [math.log(1/10,2.0)] * 10
    h_words_actual = -np.mean(log_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 

    # window_size = 1, is_logprob = False
    model = NGramModel(text, alpha=1, n=2, is_logprob=False)
    h_words_test, h_wordset_test = survey_text(model, 1)
    log_array = [1/10] * 10
    h_words_actual = np.mean(log_array)
    # print(h_words_test, h_words_actual)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 
    
    # window_size = 2, is_logprob = True
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2), window_size=2)
    plog_array = [math.log(1/9,2.0)] * 9
    h_words_actual = -np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12)

    # window_size = 2, is_logprob = False
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2, is_logprob=False), window_size=2)
    plog_array = [1/9] * 9
    h_words_actual = np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12)

    # window_size = 3, is_logprob = True
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2), window_size=3)
    plog_array = [2*math.log(1/9,2.0) - math.log(1/10,2.0)] * 8 
    h_words_actual = -np.mean(plog_array)
    print(h_words_test, h_words_actual)
    assert round(h_words_test, 12) == round(h_words_actual, 12)

    # window_size = 3, is_logprob = False
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2, is_logprob=False), window_size=3)
    plog_array = [(1/9)**2 / (1/10)] * 8 
    h_words_actual = np.mean(plog_array)
    print(h_words_test, h_words_actual)
    assert round(h_words_test, 12) == round(h_words_actual, 12)

    # window_size = 4, is_logprob = True
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2), window_size=4)
    plog_array = [3*math.log(1/9,2.0) - 2*math.log(1/10,2.0)]*7
    h_words_actual = -np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12)  \

    # window_size = 4, is_logprob = False
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2, is_logprob=False), window_size=4)
    plog_array = [(1/9)**3 / (1/10)**2]*7
    h_words_actual = np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12)  

    # window_size = 5, is_logprob = True
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2), window_size=5)
    plog_array = [4*math.log(1/9,2.0) - 3*math.log(1/10,2.0)]*6
    h_words_actual = -np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 

    # window_size = 5, is_logprob = False
    h_words_test, h_word_set_test = survey_text(NGramModel(text, alpha=1, n=2, is_logprob=False), window_size=5)
    plog_array = [(1/9)**4 / (1/10)**3]*6
    h_words_actual = np.mean(plog_array)
    assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_3gram():
  # window_size = 2 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3), window_size=2)
  plog_array1 = [math.log(1/9,2.0)]*9
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 2 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3, is_logprob=False), window_size=2)
  plog_array1 = [1/9]*9
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 3 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3), window_size=3)
  plog_array1 = [math.log(1/8,2.0)] * 8
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 3 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3, is_logprob=False), window_size=3)
  plog_array1 = [1/8] * 8
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 4 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3), window_size=4)
  plog_array1 = [2*math.log(1/8,2.0) - math.log(1/9,2.0)] * 7
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 4 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3, is_logprob=False), window_size=4)
  plog_array1 = [(1/8)**2 / (1/9)] * 7
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 5 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3), window_size=5)
  plog_array1 = [3*math.log(1/8,2.0) - 2*math.log(1/9,2.0)] * 6
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 5 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3, is_logprob=False), window_size=5)
  plog_array1 = [(1/8)**3 / (1/9)**2] * 6
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 6 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3), window_size=6)
  plog_array1 = [4*math.log(1/8,2.0) - 3*math.log(1/9,2.0)] * 5
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 6 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=3, is_logprob=False), window_size=6)
  plog_array1 = [(1/8)**4 / (1/9)**3] * 5
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

def test_4gram():
  # window_size = 8 (trigram), is_logprob = True
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=4), window_size=8)
  plog_array1 = [5*math.log(1/7,2.0) - 4*math.log(1/8,2.0)]*3
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 8 (trigram), is_logprob = False
  h_words_test, h_word_set_test = survey_text(NGramModel("a b c d e f g h i", alpha=1, n=4, is_logprob=False), window_size=8)
  plog_array1 = [(1/7)**5 / (1/8)**4]*3
  h_words_actual = np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

def test_ngrams():
  model2 = NGramModel("Hello there. My name is Ryan and I am from Los Angeles.",alpha=3, n=2)
  model3 = NGramModel("Hello there. My name is Ryan and I am from Los Angeles.",alpha=3, n=3)
  model4 = NGramModel("Hello there. My name is Ryan and I am from Los Angeles.",alpha=3, n=4)
  assert logp_words(model2, ['ryan']) == logp_words(model3, ['ryan']) == logp_words(model4, ['ryan'])

if __name__ == '__main__':
    import nose
    nose.runmodule()