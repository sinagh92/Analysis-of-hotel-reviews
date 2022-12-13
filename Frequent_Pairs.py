import nltk
import numpy as np
from collections import Counter
from itertools import combinations


def Frequent_Pairs(neg_sentences):
    """ Finding the frequent paies for the set of sentences"""
    d = Counter()
    nouns = []
    adj = []
    for i, sub in enumerate(neg_sentences):
        if len(neg_sentences) < 2:
            continue
        tokens = nltk.word_tokenize(sub)
        sentence_nouns = [u for u, val in nltk.pos_tag(tokens) if 'NN' in val]
        nouns.append(sentence_nouns)
        sentence_adjs = [u for u, val in nltk.pos_tag(tokens) if 'JJ' in val]
        adj.append(sentence_adjs)

        neg_nouns = sentence_nouns + sentence_adjs
        subs = list(set(neg_nouns))
        for comb in combinations(subs, 2):
            d[comb] += 1
    frequent_items = d.most_common()

    all_nouns = []
    all_adjs = []
    # separating adjectives and nouns
    for i in range(0, len(nouns)):
        for j in range(0, len(nouns[i])):
            all_nouns.append(nouns[i][j])
    for i in range(0, len(adj)):
        for j in range(0, len(adj[i])):
            all_adjs.append(adj[i][j])
    all_adjs = list(set(all_adjs))
    all_nouns = list(set(all_nouns))
    all_adjs = list(set(all_adjs) - set(all_nouns))

    # ordering the frequent items as noun : adjective
    ordered_items = []
    for i in range(0, len(frequent_items)):
        if frequent_items[i][0][0] in all_nouns and frequent_items[i][0][1] in all_adjs:
            ordered_items.append(frequent_items[i])
        elif frequent_items[i][0][1] in all_nouns and frequent_items[i][0][0] in all_adjs:
            noun = frequent_items[i][0][1]
            adj = frequent_items[i][0][0]
            new_tuple = ((noun, adj), frequent_items[i][1])
            ordered_items.append(new_tuple)
    # sorting frequent items based on their support
    sorted_values = []
    score_vector = []
    sorted_items = []
    for i in range(0, len(ordered_items)):
        sorted_values.append(ordered_items[i][0])
        score_vector.append(ordered_items[i][1])
    sorted_ind = np.flip(np.argsort(score_vector))
    for i in range(0, len(sorted_ind)):
        sorted_items.append(
            (sorted_values[sorted_ind[i]], score_vector[sorted_ind[i]]))
    return sorted_items
