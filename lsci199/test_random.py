from ngram_model import *

# text = "Hello there. My name is Ryan and I am from Los Angeles. I am currently an undergraduate at UCI studying Language Science. I love to listen to jazz, and I enjoy running and climbing. Nice to meet you all. Your turn."
text = moby_dick
m_rt_rw = NGramModel(text, alpha=0.1, n=3, randomize_text=True, ordered_windows=False)
m_rt_ow = NGramModel(text, alpha=0.1, n=3, randomize_text=True)
m_ot_rw = NGramModel(text, alpha=0.1, n=3, ordered_windows=False)
m_ot_ow = NGramModel(text, alpha=0.1, n=3)
# m2 = NGramModel(text, alpha=0.1, n=3, randomize_text=True)
# m3 = NGramModel(text, alpha=0.1, n=3, randomize_text=True)
# m4 = NGramModel(text, alpha=0.1, n=3, randomize_text=True)
# m5 = NGramModel(text, alpha=0.1, n=3, randomize_text=True)




text_lo_e = "1 0 1 1 1 1 0 0 1 1 0 1 1 1 1 0 0 1 0 0 0 1 1 1 0 1 0 1 1 1"
text_hi_e = "1 2 3 4 k i q w a s d f g h z x c v b n m k p o i 0 9 8 7 6"

lo_e = NGramModel(text_lo_e, alpha=0.1, n=3)
hi_e = NGramModel(text_hi_e, alpha=0.1, n=3)

def windows_1to5(model):
    # print(model.text)
    for i in range(1,6):
        h_words_current, h_wordset_current = survey_text(model, i)
        print("window_size", i, "h_words",h_words_current, "h_wordset", h_wordset_current)
    print()



