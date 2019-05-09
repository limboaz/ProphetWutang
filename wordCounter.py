#LDA analysis code for two time periods: 2010-2014, 2015-2019
from collections import Counter
from pprint import pprint
import string
import nltk
import json
import lda
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

more_stop_words = set()
more_stop_words.add("yeah")
more_stop_words.add("like")
more_stop_words.add("just")
more_stop_words.add("know")
more_stop_words.add("want")
more_stop_words.add("need")
more_stop_words.add("got")
more_stop_words.add("make")
more_stop_words.add("wanna")
more_stop_words.add("gonna")
more_stop_words.add("gotta")
more_stop_words.add("tryna")
more_stop_words.add("told")
more_stop_words.add("this")
more_stop_words.add("maybe")
more_stop_words.add("cause")
more_stop_words.add("nigga")
more_stop_words.add("niggas")
more_stop_words.add("used")
more_stop_words.add("the")

exclude = more_stop_words | set(string.punctuation)
split_it = set()
def preprocess_word():
    global exclude, split_it
    f = open("wordclouds/2010-2014.txt")
    stemmer = nltk.PorterStemmer()
    songs = ""
    for line in f.readlines():
        # split() returns list of all the words in the string
        line = line.replace("\\n", " ") 
        line = ''.join(ch for ch in line.lower() if ch not in exclude)
        line = line.lower()
        try:
            line = stemmer.stem(line)
        except:
            pass
        songs += line
        split_it = split_it | set(line.split()) 
    write_back = open("preprocessed-2010-2014.txt", "w+")
    write_back.write(songs)
    write_back.close()
    f.close()


# This part of code intended to obtained the top k words in the given corpus, but we decided to do something else. 
# Counter = Counter(split_it) 

# most_common() produces k frequently encountered 
# input values and their respective counts. 
# most_occur = Counter.most_common(300)
# for word in most_occur:
#     print(word[0]+ "
# :" +str(word[1]))

#LDA analysis code
def lda_lda(docs, num_topics, num_iters, n_top_words):
    vectorizer = CountVectorizer(max_features=1000,
                        stop_words='english',
                        lowercase=True)
    pprint(docs.read(100))
    data = vectorizer.fit_transform(docs)
    data *= 100
    data = data.astype(int)

    vocab = vectorizer.get_feature_names()

    model = lda.LDA(n_topics=num_topics, n_iter=num_iters, random_state=1)
    model.fit(data)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    ret = []

    print(model.nz_)	
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]

        ret.append(u'Topic {}: {}'.format(i, u' '.join(topic_words)).encode('utf-8').strip())
    pprint(ret)
    return ret 

preprocess_word()
lda_lda(open("preprocessed-2010-2014.txt"), 10, 10000, 10)

