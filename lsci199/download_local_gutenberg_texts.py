import os
import requests

def download_gutenberg_text(url):
  """ Download a text file from a Project Gutenberg URL and return it as a string. """
  return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

# nltk.download("punkt") # This gives us access to nltk.tokenize.word_tokenize
url_prefix = "http://www.socsci.uci.edu/~rfutrell/teaching/lsci109-w2020/data/"
pride_and_prejudice = download_gutenberg_text(url_prefix + "1342-0.txt")
two_cities = download_gutenberg_text(url_prefix + "98-0.txt")
moby_dick = download_gutenberg_text(url_prefix + "2701-0.txt")
hard_times = download_gutenberg_text(url_prefix + "786-0.txt")

mega_text_4 = pride_and_prejudice + moby_dick + two_cities + hard_times

all_files = os.listdir("books/")
mega_text_100, mega_text_90, mega_text_50, mega_text_10 = '','','',''
i=0
for file in all_files:
    path = 'books/'+file
    with open(path, 'r', encoding="utf-8", errors="ignore") as current_file:
        text = current_file.read().encode('utf-8')[1:]
        text = text.decode('utf-8', 'ignore')
        mega_text_100 += text
        if i < 88:
            mega_text_90 += text
        if i < 42:
            mega_text_50 += text
        if i < 8:
            mega_text_10 += text
    i+=1

# l10 = len(mega_text_10)
# l50 = len(mega_text_50)
# l90 = len(mega_text_90)
# l100 = len(mega_text_100)
# print(l10/l100)
# print(l50/l100)
# print(l90/l100)

