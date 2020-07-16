from ngram_model import *
import time

def special_shuffle(l):
    sufficient = False
    num_shuffles = 0
    l.remove('.')
    while sufficient is False:
        if num_shuffles == 50:
            print("try again...")
        if num_shuffles == 200:
            print("whoa man. you really wanna wait more?")
        if num_shuffles == 1000:
            print("seriously, just start over...")
        if num_shuffles == 10000:
            print("X.X")
        for i in range(len(l)-1):
            if  (l[0] == '.'):
                shuffle(l)
                num_shuffles += 1
                break
            elif (l[i] == '.' and l[i+1] == '.'):
                shuffle(l)
                num_shuffles += 1
                break
            elif l[len(l)-1] == '.':
                shuffle(l)
                num_shuffles += 1
                break 
            if i == len(l)-2:
                sufficient = True
    l.append('.')
    return l

def divide_and_conquer(l, size):
    final_arr = []
    div = int(len(l)/size)
    r = len(l) % size
    for i in range(div):
        final_arr += special_shuffle(l[i*size:(i+1)*size])
        if i == div-1:
            final_arr += special_shuffle(l[(i+1)*size:(i+1)*size+r])
    return final_arr
            

text = 'My name is Bradley, and I am the new intern. I am from Ohio and I really like to fish. Eating steak is my passion, and I can.'
# text = pride_and_prejudice
start = time.time()
m = NGramModel(text, alpha=0.5, n=3, randomize_text=True)
end = time.time()
print(end-start)
t = m.tokens_pre_randomized_text
print("len(t)",len(t))
start = time.time()

l = divide_and_conquer(t, 2000)
print(l, len(l))


end = time.time()
print(end-start)