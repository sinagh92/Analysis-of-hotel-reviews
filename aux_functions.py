from nltk import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk.corpus import stopwords
from collections import Counter
from itertools import combinations
import numpy as np


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


def get_words_for_each_cluster(Pos_reviews):
    """finding frequent words and their support in each cluster"""

    Positive_Reviews = Pos_reviews.all_reviews
    clusters_pos = Pos_reviews.clusters

    clusters_pos_words = []
    clusters_pos_supp = []
    for cluster_num in range(len(clusters_pos)):
        sentences = [Positive_Reviews[i] for i in clusters_pos[cluster_num]]
        sentences = stop_word_remover(sentences)
        [w_pos, s_pos] = find_frequent_items(sentences)

        sort_i_p = np.flip(np.argsort(s_pos))
        w_pos = [w_pos[i] for i in sort_i_p]
        s_pos = [s_pos[i] for i in sort_i_p]
        clusters_pos_words.append(w_pos)
        clusters_pos_supp.append(s_pos)

    Pos_reviews.clusters_words = clusters_pos_words
    Pos_reviews.clusters_supp = clusters_pos_supp


def compute_algorithm_score(Pos_reviews, Neg_reviews):
    """
    This function computes the score for the proposed algorithm

    Parameters
    ----------
    Pos_reviews : TYPE Reviews object
        DESCRIPTION. for positive reviews
    Neg_reviews : TYPE Reviews object 
        DESCRIPTION. for negative reviews

    Returns
    -------
    common_words_count : TYPE int
        DESCRIPTION. number of common words

    """
    # computing the algorithm score

    clusters_pos = Pos_reviews.clusters
    clusters_neg = Neg_reviews.clusters
    clusters_pos_words = Pos_reviews.clusters_words
    clusters_neg_words = Neg_reviews.clusters_words

    temp = 0
    # for positive clusters

    for i in range(0, len(clusters_pos)):
        w_pos_1 = clusters_pos_words[i][0:5]
        for j in range(i+1, len(clusters_pos)):
            w_pos_2 = clusters_pos_words[j][0:5]
            common1 = set(w_pos_1) - set(w_pos_2)
            common2 = set(w_pos_2) - set(w_pos_1)
            union = set(w_pos_1 + w_pos_2)
            common = union - common1 - common2
            temp = len(common) + temp

    # for Negative clusters
    for i in range(0, len(clusters_neg)):
        w_neg_1 = clusters_neg_words[i][0:5]
        for j in range(i+1, len(clusters_neg)):
            w_neg_2 = clusters_neg_words[j][0:5]
            common1 = set(w_neg_1) - set(w_neg_2)
            common2 = set(w_neg_2) - set(w_neg_1)
            common = set(w_neg_1 + w_neg_2) - common1 - common2
            temp = len(common) + temp
    common_words_count = temp
    return common_words_count


def save_important_sentences(Neg_reviews, Pos_reviews):
    """
    This function computes and stores the important sentences for each cluster

    Parameters
    ----------
    Neg_reviews : TYPE Reviews object
        DESCRIPTION. for negative reviews
    Pos_reviews : TYPE Reviews object
        DESCRIPTION. for positive reviews

    Returns
    -------
    None.

    """
    Negative_Reviews = Neg_reviews.all_reviews

    clusters_neg = Neg_reviews.clusters
    Positive_Reviews = Pos_reviews.all_reviews
    clusters_pos = Pos_reviews.clusters
    clusters_neg_words = Neg_reviews.clusters_words
    clusters_neg_supp = Neg_reviews.clusters_supp
    clusters_pos_words = Pos_reviews.clusters_words
    clusters_pos_supp = Pos_reviews.clusters_supp

    # initialize vectors for saving important sentences
    sentence_score_neg = [
        [0 for i in range(len(Negative_Reviews))] for cluster_num in clusters_neg]
    sentence_score_pos = [
        [0 for i in range(len(Negative_Reviews))] for cluster_num in clusters_pos]
    cluster_sentence_neg = [[[]
                             for i in range(5)] for cluster_num in clusters_neg]
    cluster_score_neg = [[0 for i in range(5)] for cluster_num in clusters_neg]
    cluster_sentence_pos = [[[]
                             for i in range(5)] for cluster_num in clusters_pos]
    cluster_score_pos = [[0 for i in range(5)] for cluster_num in clusters_pos]
    # finding the most important sentences
    for cluster_num in range(len(clusters_neg)):
        w_neg = clusters_neg_words[cluster_num]
        w_sup = clusters_neg_supp[cluster_num]
        w_neg_dic = dict(zip(w_neg, w_sup))

        # This part was for finding the most length of reviews to use in
        # review score formula. However, we did not use it for final experiment
        # since it didnt work well.
        ### max_len_rev = 0
        # for index, sentence in enumerate(clusters_neg[cluster_num]):
        ###     rev = Negative_Reviews[sentence]
        ###     temp = len(rev)
        # if temp > max_len_rev:
        ###         max_len_rev = temp

        for index, sentence in enumerate(clusters_neg[cluster_num]):
            rev = Negative_Reviews[sentence]
            rev = rev.split()
            score = 0

            for i in range(0, len(w_neg)):
                if w_neg[i] in rev:
                    score = score + w_neg_dic[w_neg[i]]
            sentence_score_neg[cluster_num][sentence] = score/(len(rev)+1)
        sorted_sentence_score_ind = np.flip(
            np.argsort(sentence_score_neg[cluster_num]))
        for i in range(0, 5):
            cluster_score_neg[cluster_num][i] = sentence_score_neg[cluster_num][sorted_sentence_score_ind[i]]
            cluster_sentence_neg[cluster_num][i] = Negative_Reviews[sorted_sentence_score_ind[i]]

    for cluster_num in range(len(clusters_pos)):
        w_pos = clusters_pos_words[cluster_num]
        w_sup = clusters_pos_supp[cluster_num]
        max_w_sup = max(w_sup)
        w_pos_dic = dict(zip(w_pos, w_sup))
        max_len_rev = 0
        # for index, sentence in enumerate(clusters_pos[cluster_num]):
        ###     rev = Negative_Reviews[sentence]
        ###     temp = len(rev)
        # if temp > max_len_rev:
        ###         max_len_rev = temp
        for index, sentence in enumerate(clusters_pos[cluster_num]):
            rev = Positive_Reviews[sentence]
            rev = rev.split()
            score = 0

            for i in range(0, len(w_pos)):
                if w_pos[i] in rev:
                    score = score + w_pos_dic[w_pos[i]]
            sentence_score_pos[cluster_num][sentence] = score/(len(rev)+1)
        sorted_sentence_score_ind = np.flip(
            np.argsort(sentence_score_pos[cluster_num]))
        for i in range(0, 5):
            cluster_score_pos[cluster_num][i] = sentence_score_pos[cluster_num][sorted_sentence_score_ind[i]]
            cluster_sentence_pos[cluster_num][i] = Positive_Reviews[sorted_sentence_score_ind[i]]

    Neg_reviews.sentence_score = sentence_score_neg
    Pos_reviews.sentence_score = sentence_score_pos
    Neg_reviews.cluster_sentence = cluster_sentence_neg
    Neg_reviews.cluster_score = cluster_score_neg
    Pos_reviews.cluster_sentence = cluster_sentence_pos
    Pos_reviews.cluster_score = cluster_score_pos


def write_results(Pos_reviews, Neg_reviews, name, nb_clusters, nb_comments, hotel_count, Hotel_name, write_freq_items=False, write_freq_pairs=False):
    """
    This functions writes the results in record file. To write the frequent items and frequent pairs, write_freq_items and write_freq_pairs should be True.

    Parameters
    ----------
    Pos_reviews : TYPE Reviews object
        DESCRIPTION. for positive reviews
    Neg_reviews : TYPE Reviews object
        DESCRIPTION. for negative reviews
    name : TYPE string
        DESCRIPTION. name of the file
    nb_clusters : TYPE
        DESCRIPTION.
    nb_comments : TYPE integer
        DESCRIPTION. number of comments
    hotel_count : TYPE integer
        DESCRIPTION. number of hotels
    Hotel_name : TYPE string
        DESCRIPTION. name of the hotel being processed
    write_freq_items : TYPE, optional bool
        DESCRIPTION. The default is False.
    write_freq_pairs : TYPE, optional bool
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    all_sentences_neg = Neg_reviews.all_sentences
    all_sentences_pos = Pos_reviews.all_sentences

    cluster_score_neg = Neg_reviews.cluster_score
    cluster_score_pos = Pos_reviews.cluster_score
    clusters_pos = Pos_reviews.clusters

    clusters_neg = Neg_reviews.clusters
    clusters_neg_words = Neg_reviews.clusters_words
    clusters_neg_supp = Neg_reviews.clusters_supp
    clusters_pos_words = Pos_reviews.clusters_words
    clusters_pos_supp = Pos_reviews.clusters_supp
    cluster_sentence_neg = Neg_reviews.cluster_sentence
    cluster_sentence_pos = Pos_reviews.cluster_sentence
    words_neg = Neg_reviews.words
    words_pos = Pos_reviews.words

   # writing the results in record file
    record_file = open("results/"+name + "_" + str(nb_clusters) + "_" +
                       str(nb_comments) + "/" + str(hotel_count) + "_" + Hotel_name + ".txt", "w")
    record_file.write("Negative Comments Clusters\n")
    for cluster_num in range(len(clusters_neg)):
        record_file.write("===================\n")
        record_file.write("cluster " + str(cluster_num) + ":" + "\n")
        w_neg = clusters_neg_words[cluster_num]
        w_sup = clusters_neg_supp[cluster_num]
        for i in range(0, min(10, len(w_neg))):
            record_file.write(w_neg[i] + " : " + str(w_sup[i]) + "\t")
        record_file.write("\n")
        record_file.write("--------------------\n")
        for i in range(0, 5):
            record_file.write("score " + str(cluster_score_neg[cluster_num][i]) + ": " + str(
                cluster_sentence_neg[cluster_num][i]) + "\n")
            record_file.write("--------------------\n")

    record_file.write("===================\n")
    record_file.write("===================\n")
    record_file.write("===================\n")
    record_file.write("Positive Comments Clusters\n")
    for cluster_num in range(len(clusters_pos)):
        record_file.write("===================\n")
        record_file.write("cluster " + str(cluster_num) + ":" + "\n")
        w_pos = clusters_pos_words[cluster_num]
        w_sup = clusters_pos_supp[cluster_num]
        for i in range(0, min(10, len(w_pos))):
            record_file.write(w_pos[i] + " : " + str(w_sup[i]) + "\t")
        record_file.write("\n")
        record_file.write("--------------------\n")
        for i in range(0, 5):
            record_file.write("score " + str(cluster_score_pos[cluster_num][i]) + ": " + str(
                cluster_sentence_pos[cluster_num][i]) + "\n")
            record_file.write("--------------------\n")
    if write_freq_items == True or write_freq_pairs == True:
        # We also found frequent pairs in all sentences
        # which is not used in our final experiments
        # finding all frequent items of a hotel review ( not used in final experiment)

        if len(all_sentences_neg) > 0:
            [words_neg, support_neg] = find_frequent_items(all_sentences_neg)
            pairs_neg = Frequent_Pairs(all_sentences_neg)

        if len(all_sentences_pos) > 0:
            [words_pos, support_pos] = find_frequent_items(all_sentences_pos)
            pairs_pos = Frequent_Pairs(all_sentences_pos)

        dic_neg = dict(zip(words_neg, support_neg))
        dic_pos = dict(zip(words_pos, support_pos))

        # pairs processing:
        words = []
        adjs = []
        for i in range(0, len(pairs_neg)):
            words.append(pairs_neg[i][0][0])
            adjs.append(pairs_neg[i][0][1])
        words = list(set(words))
        adjs = [[] for i in range(0, len(words))]
        words_dic = dict(zip(words, adjs))
        for i in range(0, len(pairs_neg)):
            words_dic[pairs_neg[i][0][0]].append(pairs_neg[i][0][1])
            words_dic[pairs_neg[i][0][0]].append(str(pairs_neg[i][1]))
        pairs_neg_dict = words_dic

        words = []
        adjs = []
        for i in range(0, len(pairs_pos)):
            words.append(pairs_pos[i][0][0])
            adjs.append(pairs_pos[i][0][1])
        words = list(set(words))
        adjs = [[] for i in range(0, len(words))]
        words_dic = dict(zip(words, adjs))
        for i in range(0, len(pairs_pos)):
            words_dic[pairs_pos[i][0][0]].append(pairs_pos[i][0][1])
            words_dic[pairs_pos[i][0][0]].append(str(pairs_pos[i][1]))
        pairs_pos_dict = words_dic

        # removing words that effect the results most and stop words
        removing_words = ['hotel', 'hotels', 'also', 'would', 'could']
        words_neg = [
            t for t in words_neg if t not in stopwords.words('english')]
        words_pos = [
            t for t in words_pos if t not in stopwords.words('english')]
        words_neg = [t for t in words_neg if t not in removing_words]
        words_pos = [t for t in words_pos if t not in removing_words]

        supp_neg = [[] for i in range(0, len(words_neg))]
        for i, word in enumerate(words_neg):
            supp_neg[i] = dic_neg[words_neg[i]]
        supp_pos = [[] for i in range(0, len(words_pos))]
        for i, word in enumerate(words_pos):
            supp_pos[i] = dic_pos[words_pos[i]]
    if write_freq_items == True:
        # writing frequent items of all reviews in the files
        # NOT used in final experiment
        # sort frequent items with respect to their support
        sorted_inds_neg = np.flip(np.argsort(supp_neg))
        sorted_inds_pos = np.flip(np.argsort(supp_pos))
    # not used in final experiments
        record_file.write("===================\n")
        record_file.write("===Frequent Items==\n")
        record_file.write("===================\n")
        record_file.write("======Negative=====\n")
        for i in range(0, min(10, len(words_neg))):
            record_file.write(
                words_neg[sorted_inds_neg[i]] + " : " + str(supp_neg[sorted_inds_neg[i]]) + "\n")
        record_file.write("======Positive=====\n")
        for i in range(0, min(10, len(words_pos))):
            record_file.write(
                words_pos[sorted_inds_pos[i]] + " : " + str(supp_pos[sorted_inds_pos[i]]) + "\n")

    if write_freq_pairs == True:
        # writing frequent pairs in the file
        record_file.write("===================\n")
        record_file.write("===Frequent Pairs==\n")
        record_file.write("===================\n")
        record_file.write("======Negative=====\n")
        neg_keys = list(pairs_neg_dict.keys())
        for i in range(0, min(10, len(neg_keys))):
            record_file.write(neg_keys[i] + " : " + "\t")
            adjs = pairs_neg_dict[neg_keys[i]]
            for j in range(0, len(adjs)):
                record_file.write(adjs[j] + "\t")
            record_file.write("\n")
            record_file.write("--------------------\n")
        record_file.write("======Positive=====\n")
        pos_keys = list(pairs_pos_dict.keys())
        for i in range(0, min(10, len(pos_keys))):
            record_file.write(pos_keys[i] + " : " + "\t")
            adjs = pairs_pos_dict[pos_keys[i]]
            for j in range(0, len(adjs)):
                record_file.write(adjs[j] + "\t")
            record_file.write("\n")
            record_file.write("--------------------\n")

    record_file.write("=====Summary=====\n")
    record_file.write("=====Positive====\n")
    for cluster_num in range(len(clusters_pos)):
        for i in range(0, 1):
            record_file.write(str(cluster_num) + ": " +
                              str(cluster_sentence_pos[cluster_num][i]) + "\n")
    record_file.write("=====Negative====\n")
    for cluster_num in range(len(clusters_neg)):
        for i in range(0, 1):
            record_file.write(str(cluster_num) + ": " +
                              str(cluster_sentence_neg[cluster_num][i]) + "\n")


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
    tokens = [stemmer.stem(t)
              for t in tokens if t not in stopwords.words('english')]
    return tokens


def stop_word_remover(text):
    # tokenizes and stems the text'
    removing_words = ['hotel', 'hotels', 'also', 'would', 'could']
    text2 = [[] for i in range(0, len(text))]
    for i in range(0, len(text)):
        sentence = text[i].lower()
        sentence = sentence.replace('rooms', 'room')
        tokens = word_tokenize(sentence)
        tokens = [word for word in tokens if len(
            word) > 2 and word not in removing_words]
        text2[i] = " ".join(
            word for word in tokens if word not in stopwords.words('english'))
    return text2
