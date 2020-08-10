import os
import requests

text = ''
with open('chinese_corpus_size_small', 'r', encoding="utf-8") as current_file:
    text = current_file.read().encode('utf-8')[1:]
    text = text.decode('utf-8', 'ignore')
    l = len(text)
    text = text[:int(l/1000)]

chinese_full = text
print("Chinese text loaded!")

file = open("chinese_corpus_size_smaller", "w")
file.write(chinese_full)
print("DONE")

# chinese_90 = text[:int(l*.9)] 
# chinese_10 = text[int(l*.9):]
# turkish_90 = text[:int(l*.9/5)] 
# turkish_10 = text[int(l*.9):]

# with open('enwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
#     text = current_file.read().encode('utf-8')[1:]
#     text = text.decode('utf-8', 'ignore')
#     # text = text[:int(len(text))]
#     l = len(text)

# english_full = text
# print("English text loaded!")
# print(len(english_full))


# text = ''
# with open('dutch_corpus_size_small', 'r', encoding="utf-8") as current_file:
#     text = current_file.read().encode('utf-8')[1:]
#     text = text.decode('utf-8', 'ignore')
#     l = len(text)
#     text = text[:int(l/25)]

# dutch_full = text
# print("Dutch text loaded!")

# file = open("dutch_corpus_size_small", "w")
# file.write(dutch_full)
# print("DONE")
