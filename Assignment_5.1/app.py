import re
import math
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlopen
from bs4 import BeautifulSoup
from collections import defaultdict, OrderedDict
import itertools

urls = [
        "https://futureoflife.org/background/benefits-risks-of-artificial-intelligence/",
        "https://builtin.com/artificial-intelligence",
        "https://www.sas.com/en_us/insights/analytics/what-is-artificial-intelligence.html",
        "https://www.brookings.edu/research/what-is-artificial-intelligence/",
        "https://www.accenture.com/ph-en/insights/artificial-intelligence-summary-index",
        "https://plato.stanford.edu/entries/artificial-intelligence/",
        "https://sitn.hms.harvard.edu/flash/2017/history-artificial-intelligence/",
        "https://artificialintelligence-news.com/2019/08/14/google-project-euphonia-voice-recognition/",
        "https://www.mygreatlearning.com/blog/what-is-artificial-intelligence/",
        "https://www.cigionline.org/articles/cyber-security-battlefield/?utm_source=google_ads&utm_medium=grant&gclid=CjwKCAjwgb6IBhAREiwAgMYKRm6JMR_GYAfLyKKm3kJvaRUm5RpiZjtZ6TcP_igiieXQm-3GZ5XN4hoC88YQAvD_BwE",
]

def word_list(urls):

    all_words = []
    
    for url in urls:
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
        for script in soup(["script", "style"]):
            script.extract()  
        text = soup.get_text()
        text = re.sub(r'[^\w]', ' ', text)
        words = text.split(' ')
        words = [x for x in words if x]
        all_words.append(words)

    return all_words

def list_struct(): return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

all_words = word_list(urls)

words_tf = defaultdict(list_struct)
words_idf = {}
words_tf_idf = defaultdict(list_struct)
words_tf_idf_sum = {}

for i in range(len(all_words)):
    words = all_words[i]
    n = len(words)
    for word in words:
        tf_list = words_tf[word]
        tf_list[i] = ((tf_list[i] * n) + 1) / n
        words_tf[word] = tf_list

for key in words_tf.keys():
    zero = words_tf[key].count(0)
    f = 10 - zero

    words_idf[key] = math.log10(10/f)

for key in words_tf.keys():
    Sum = 0.0
    for i in range(len(words_tf[key])):
        tf = words_tf[key][i]
        score = tf * words_idf[key]
        Sum += score
        words_tf_idf[key][i] = score
    words_tf_idf_sum[key] = Sum

words_tf_idf_sum = dict(sorted(words_tf_idf_sum.items(), key=lambda item: item[1], reverse=True))

sliced_words_tf_idf_sum = dict(itertools.islice(words_tf_idf_sum.items(), 10))


x = list(sliced_words_tf_idf_sum.keys())
y = list(sliced_words_tf_idf_sum.values())

ind = np.arange(len(y))

fig, ax = plt.subplots()
fig.set_size_inches(18.5, 10.5)
ax.barh(ind, y)
ax.set_yticks(ind)
ax.set_yticklabels(x)

ax.bar_label(ax.containers[0])
plt.gca().invert_yaxis()
plt.title("Top 10 Most Significant Words in 10 Websites")
plt.show()