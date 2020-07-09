from h_wordorder import *

def test_bigram():
  # window_size = 1
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=1)
  words_plog = [math.log(1/12,2.0)]*12
  h_words_actual = -np.mean(words_plog)
  print(h_words_actual, h_words_test)
  # assert round(h_words_test, 12) == round(h_words_actual, 12) 
  assert h_words_test == h_words_actual 

  # window_size = 2
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=2)
  plog_array = [math.log(1/11,2.0)] * 10
  h_words_actual = -np.mean(plog_array)
  # assert round(h_words_test, 12) == round(h_words_actual, 12)
  assert h_words_test == h_words_actual 

  # window_size = 3
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=3)
  plog_array = [math.log(1/12,2.0) + math.log(1/11,2.0) - math.log(1/12,2.0) + math.log(1/11,2.0) -math.log(1/12,2.0)] * 8 
  h_words_actual = -np.mean(plog_array)
  print(h_words_test, h_words_actual)
  # assert round(h_words_test, 12) == round(h_words_actual, 12)
  assert h_words_test == h_words_actual 

  # window_size = 4
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=4)
  plog_array = [math.log(1/12,2.0) + 3*(math.log(1/11,2.0) - math.log(1/12,2.0))]*7
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12)  
 
  # window_size = 5
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles."), window_size=5)
  plog_array = [math.log(1/12,2.0) + 4*(math.log(1/11,2.0) - math.log(1/12,2.0))]*6
  h_words_actual = -np.mean(plog_array)
  assert round(h_words_test, 12) == round(h_words_actual, 12) 

def test_trigram():
  # window_size = 3 (trigram)
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=3)
  plog_array1 = [math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)+math.log(1/10,2.0)-math.log(1/11,2.0)] * 8
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  # window_size = 2 (trigram) 
  h_words_test, h_word_set_test = survey_text(AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3), window_size=2)
  plog_array1 = [math.log(1/11,2.0)] + [math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)]*9
  h_words_actual = -np.mean(plog_array1)
  assert round(h_words_test, 12) == round(h_words_actual, 12)

  model = AdditiveSmoothingModel("Hello, there. My name is Ryan and I am from Los Angeles.", n=3)
  test = model.total_prob(["my","name","is","ryan"])
  actual = math.log(1/12,2.0)+math.log(1/11,2.0)-math.log(1/12,2.0)+2*(math.log(1/10,2.0)-math.log(1/11,2.0))
  assert test == actual

def test_4gram():
  model = AdditiveSmoothingModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  test = model.total_prob(['my', 'name', 'is', 'ryan'])
  actual = math.log(1/9,2.0)
  assert test == actual 

  test = model.total_prob(['my','name','is','ryan','and'])
  actual = 2*math.log(1/9,2.0) - math.log(1/10,2.0)
  assert test == actual 

  test = model.total_prob(['my','name'])
  actual = math.log(1/11,2.0)
  assert test == actual

  w = get_windows(model, 4)
  assert w == [['my', 'name', 'is', 'ryan'],['name', 'is', 'ryan', 'and'],['is', 'ryan', 'and', 'i'],['ryan', 'and', 'i', 'am'],['and', 'i', 'am', 'from'],['i', 'am', 'from', 'los'],['am', 'from', 'los', 'angeles']]

def test_ngrams():
  model2 = AdditiveSmoothingModel("Hello there. My name is Ryan and I am from Los Angeles.", n=2)
  model3 = AdditiveSmoothingModel("Hello there. My name is Ryan and I am from Los Angeles.", n=3)
  model4 = AdditiveSmoothingModel("Hello there. My name is Ryan and I am from Los Angeles.", n=4)
  assert logp_words(model2, ['ryan']) == logp_words(model3, ['ryan']) == logp_words(model4, ['ryan'])

def test_windows():
  model = AdditiveSmoothingModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=5)
  assert len(windows) == 9

  model = AdditiveSmoothingModel("There once was a frog. He lived in a bog. And he played his fiddle in the middle of a puddle. What a muddle.", n=2)
  windows = get_windows(model, window_size=7)
  assert len(windows) == 5

if __name__ == '__main__':
    import nose
    nose.runmodule()