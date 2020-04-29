from h_wordorder import *

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

# nltk.download("punkt") # This gives us access to nltk.tokenize.word_tokenize
url_prefix = "http://www.socsci.uci.edu/~rfutrell/teaching/lsci109-w2020/data/"
pride_and_prejudice = download_gutenberg_text(url_prefix + "1342-0.txt")
# two_cities = download_gutenberg_text(url_prefix + "98-0.txt")
# moby_dick = download_gutenberg_text(url_prefix + "2701-0.txt")
# hard_times = download_gutenberg_text(url_prefix + "786-0.txt")
text = pride_and_prejudice 
model = AdditiveSmoothingNGramModel(text, n=2)
print(survey_text(model, model.tokens, 3))
