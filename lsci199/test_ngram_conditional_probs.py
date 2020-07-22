from h_wordorder import *

def test_2grams():
    text = "Hi. Hello there. My name is really Ryan Kwangmoo Lee."
    m = NGramModel(text, alpha=0, n=2)
    
    
def test_3gram():

if __name__ == '__main__':
    import nose
    nose.runmodule()