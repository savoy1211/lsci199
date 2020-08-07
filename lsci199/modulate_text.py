import nltk
import jieba
from random import shuffle
# from download_local_gutenberg_texts import *

class ModulateText:
    def __init__(self, text, randomize_within_sentence=False, randomize_across_sentence=False, language="english"):
        self.text = text

        if language == "english":
            self.tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
            self.state = "ordered"

            if randomize_within_sentence:
                sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
                sentences = sent_detector.tokenize(text.strip(), realign_boundaries=False)
                final_randomized_tokens = []
                for sentence in sentences:
                  sentence_tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(sentence) if token.isalnum()]
                  shuffle(sentence_tokens)
                  final_randomized_tokens += sentence_tokens+['.']
                text_randomized = " ".join(final_randomized_tokens)
                self.random_tokens = final_randomized_tokens
                self.state = "random within sentence"
                
            if randomize_across_sentence:
                tokens = [token.casefold() for token in nltk.tokenize.word_tokenize(text) if token.isalnum()]
                shuffle(tokens)
                self.random_tokens = tokens
                self.state = "random across sentence"

        elif language == "chinese by character":
            tokens = [token for token in jieba.cut(text, cut_all=True)]
            tokens = [str(token) for token in tokens]
            self.tokens = tokens
            self.state = "ordered"

            if randomize_across_sentence:
                shuffle(tokens)
                self.tokens = tokens
                self.state = "random across sentence"

        elif language == "chinese by word":
            tokens = [token for token in jieba.cut(text, cut_all=False)]
            tokens = [str(token) for token in tokens]
            self.tokens = tokens
            self.state = "ordered"

            if randomize_across_sentence:
                shuffle(tokens)
                self.tokens = tokens
                self.state = "random across sentence"
            
            
    
                
