import os
import requests

text = ''
with open('trwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
    text = current_file.read().encode('utf-8')[1:]
    text = text.decode('utf-8', 'ignore')
    l = len(text)
    text = text[:int(len(text)/5)]

turkish_90 = text[:int(l*.9/5)] 
turkish_10 = text[int(l*.9):]

with open('zhwiki-20181001-corpus.xml.txt', 'r', encoding="utf-8") as current_file:
    text = current_file.read().encode('utf-8')[1:]
    text = text.decode('utf-8', 'ignore')
    l = len(text)
    text = text[:int(len(text)/5)]

chinese_90 = text[:int(l*.9/5)] 
chinese_10 = text[int(l*.9):]
