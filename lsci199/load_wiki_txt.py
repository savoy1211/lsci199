import os
import requests

# def download_text(url):
#   """ Download a text file from a URL and return it as a string. """
#   return requests.get(url).content.decode('utf-8')[1:] # Remove the weird initial character

# english_full = download_text('http://langproc.socsci.uci.edu/wikitext/enwiki-20181001-corpus.xml.txt')
# print('English download complete!')
# english_train = english_full[:int(len(english_full)*.9)]
# english_test = english_full[int(len(english_full)*.9):]

text = ''
with open('enwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
    text = current_file.read().encode('utf-8')[1:]
    text = text.decode('utf-8', 'ignore')
    l = len(text)
    # text = text[:int(len(text)/5)]

english_full = text
print('English download complete!')
# english_train = text[:int(l*.9)] 
# english_test = text[int(l*.9):]

# text = ''
# with open('trwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
#     text = current_file.read().encode('utf-8')[1:]
#     text = text.decode('utf-8', 'ignore')
#     l = len(text)
#     text = text[:int(len(text)/5)]

# turkish_90 = text[:int(l*.9/5)] 
# turkish_10 = text[int(l*.9):]

# with open('zhwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
#     text = current_file.read().encode('utf-8')[1:]
#     text = text.decode('utf-8', 'ignore')
#     l = len(text)
#     text = text[:int(len(text)/5)]

# chinese_90 = text[:int(l*.9/5)] 
# chinese_10 = text[int(l*.9):]
