from h_wordorder import *

url_prefix = "http://www.socsci.uci.edu/~rfutrell/teaching/lsci109-w2020/data/"
pride_and_prejudice = download_gutenberg_text(url_prefix + "1342-0.txt")
# two_cities = download_gutenberg_text(url_prefix + "98-0.txt")
# moby_dick = download_gutenberg_text(url_prefix + "2701-0.txt")
# hard_times = download_gutenberg_text(url_prefix + "786-0.txt")
text = pride_and_prejudice 

get_h_wordorder(text)