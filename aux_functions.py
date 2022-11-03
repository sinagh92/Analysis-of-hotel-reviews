from nltk import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from collections import Counter
from itertools import combinations


def find_frequent_items(sentences):
    d = Counter()
    for i, sub in enumerate(sentences):
        if len(sentences) < 2:
            continue
        tokens = nltk.word_tokenize(sub)
        sentence_nouns = [u for u, val in nltk.pos_tag(tokens) if 'NN' in val]
        sentence_adjs = [u for u, val in nltk.pos_tag(tokens) if 'JJ' in val]
        nouns = sentence_nouns + sentence_adjs
        subs = list(set(nouns))
        for comb in combinations(subs, 1):
            d[comb] += 1
    frequent_tuples = d.most_common()
    words = []
    scores = []
    for item in frequent_tuples:
        words.append(item[0][0])
        scores.append(item[1])
    return [words, scores]


def split_uppercase(str):
    x = ''
    i = 0
    for c in str:
        if i == 0:
            x += c
        elif c.isupper() and not str[i-1].isupper():
            if i+1 < len(str) and not str[i+1].isupper():
                x += '.%s' % c
        else:
            x += c
        i += 1
    return x.strip()


def word_tokenizer(text):
    # tokenizes and stems the text
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens


def stop_word_remover(text):
    # tokenizes and stems the text'
    removing_words = ['hotel', 'hotels', 'also', 'would', 'could']
    text2 = [[] for i in range(0,len(text))]
    for i in range(0, len(text)):
        sentence = text[i].lower()
        sentence = sentence.replace('rooms', 'room')
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if len(word)>2 and word not in removing_words]
        text2[i] = " ".join(word for word in tokens if word not in stopwords.words('english'))
    return text2



