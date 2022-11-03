from sklearn.cluster import KMeans
import time
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn import cluster
import os
import warnings
from aux_functions import find_frequent_items,stop_word_remover
from cluster_sentences import cluster_sentences
from Frequent_Pairs import Frequent_Pairs

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


# https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe
# Dataset 

if __name__ == "__main__":

    # initializing required parameters
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # reading the dataset from the file
    data = pd.read_csv('Hotel_Reviews.csv')
    # replacing United Kingdom with UK to have one word for each country name
    data.Hotel_Address = data.Hotel_Address.str.replace('United Kingdom', 'UK')
    # adding the country field to data
    data['country'] = data.Hotel_Address.apply(lambda x: x.split(' ')[-1])
    # print number of reviews for all hotels
    dims = len(data)
    print(dims)
    # print number of unique hotels
    num_unique_hotels = data.Hotel_Name.nunique()
    unique_hotels = list(set(data.Hotel_Name))
    unique_hotels.sort()
    print(num_unique_hotels)
    # we use hotel data which have more than nb_comments reviews
    nb_comments = 5000
    # for loop to compute everything for 5 to 15 clusters
    for nb_clusters in [5, 15]:
        # initialize clustering algorithms
        birch = cluster.Birch(n_clusters=nb_clusters)
        kmeans = KMeans(n_clusters=nb_clusters)
        clustering_algorithms = (
            ('Birch', birch),
            ('Kmeans', kmeans)
        )
        # open file to write the results for each experiment
        results = open("clusters_" + str(nb_clusters) + "_" + str(nb_comments) + ".tsv", "w")

        for name, algorithm in clustering_algorithms:
            if not os.path.exists(name + "_" + str(nb_clusters) + "_" + str(nb_comments)):
                os.mkdir(name + "_" + str(nb_clusters) + "_" + str(nb_comments))
                print("Directory ", name, " Created ")
            else:
                print("Directory ", name, " already exists")
            # write the name of algorithm on top of data in record file
            results.write("Hotel Name" + "\t" + "reviews" + "\t" + "time" + "\t" + name + "\n")
            # open another file for each algorithm to write more detailed results
            algorithm_file = open(name + "_" + str(nb_clusters) + "_" + str(nb_comments) + "/" + "report.txt", "w")
            hotel_count = 0

            for Hotel_name in unique_hotels:
                time_start = time.clock()
                all_sentences_neg = []
                all_sentences_pos = []
                all_reviews_neg = []
                all_reviews_pos = []

                # only for one hotel, find all sentences in all reviews and
                # organize them as a list.
                Number_of_Reviews = list(data.Total_Number_of_Reviews[data.Hotel_Name == Hotel_name])[0]
                # Only for hotels which have more than nb_comments (5000) comments
                if Number_of_Reviews > nb_comments:
                    Negative_Reviews = data.Negative_Review[(data.Hotel_Name == Hotel_name)]
                    Negative_Reviews = list(Negative_Reviews)
                    Positive_Reviews = data.Positive_Review[(data.Hotel_Name == Hotel_name)]
                    Positive_Reviews = list(Positive_Reviews)

                    ### separating each sentence in each review
                    ### this part is not used in final experiments
                    # for review in Negative_Reviews:
                    #     sentences = split_uppercase(review).split('.')
                    #     all_reviews_neg.append(review.lower())
                    #     for sentence in sentences:
                    #         if len(sentence) < 10:
                    #             sentences.remove(sentence)
                    #         elif sentence != "Negative":
                    #             all_sentences_neg.append(sentence.lower())
                    #
                    # for review in Positive_Reviews:
                    #     sentences = split_uppercase(review).split('.')
                    #     all_reviews_pos.append(review.lower())
                    #     for sentence in sentences:
                    #         if len(sentence) < 10:
                    #             sentences.remove(sentence)
                    #         elif sentence != "Positive":
                    #             all_sentences_pos.append(sentence.lower())

                    print("Number of reviews : " + str(Number_of_Reviews) + " for Hotel : " + Hotel_name)
                    results.write(Hotel_name + "\t" + str(Number_of_Reviews) + "\t")

                    ### We also found frequent pairs in all sentences
                    ### which is not used in our final experiments
                    ### finding all frequent items of a hotel review ( not used in final experiment)

                    ### if len(all_sentences_neg) > 0:
                    ### [words_neg, support_neg] = find_frequent_items(all_sentences_neg)
                    ###     pairs_neg = Frequent_Pairs(all_sentences_neg)

                    ### if len(all_sentences_pos) > 0:
                    ### [words_pos, support_pos] = find_frequent_items(all_sentences_pos)
                    ###     pairs_pos = Frequent_Pairs(all_sentences_pos)

                    ### dic_neg = dict(zip(words_neg, support_neg))
                    ### dic_pos = dict(zip(words_pos, support_pos))


                    ### pairs processing:
                    ### words = []
                    ### adjs = []
                    ### for i in range(0, len(pairs_neg)):
                    ###     words.append(pairs_neg[i][0][0])
                    ###     adjs.append(pairs_neg[i][0][1])
                    ### words = list(set(words))
                    ### adjs = [[] for i in range(0, len(words))]
                    ### words_dic = dict(zip(words, adjs))
                    ### for i in range(0, len(pairs_neg)):
                    ###     words_dic[pairs_neg[i][0][0]].append(pairs_neg[i][0][1])
                    ###     words_dic[pairs_neg[i][0][0]].append(str(pairs_neg[i][1]))
                    ### pairs_neg_dict = words_dic

                    ### words = []
                    ### adjs = []
                    ### for i in range(0, len(pairs_pos)):
                    ###     words.append(pairs_pos[i][0][0])
                    ###     adjs.append(pairs_pos[i][0][1])
                    ### words = list(set(words))
                    ### adjs = [[] for i in range(0, len(words))]
                    ### words_dic = dict(zip(words, adjs))
                    ### for i in range(0, len(pairs_pos)):
                    ###     words_dic[pairs_pos[i][0][0]].append(pairs_pos[i][0][1])
                    ###     words_dic[pairs_pos[i][0][0]].append(str(pairs_pos[i][1]))
                    ### pairs_pos_dict = words_dic

                    ### removing words that effect the results most and stop words
                    ### removing_words = ['hotel', 'hotels', 'also', 'would', 'could']
                    ### words_neg = [t for t in words_neg if t not in stopwords.words('english')]
                    ### words_pos = [t for t in words_pos if t not in stopwords.words('english')]
                    ### words_neg = [t for t in words_neg if t not in removing_words]
                    ### words_pos = [t for t in words_pos if t not in removing_words]

                    ### supp_neg = [[] for i in range(0, len(words_neg))]
                    ### for i, word in enumerate(words_neg):
                    ###     supp_neg[i] = dic_neg[words_neg[i]]
                    ### supp_pos = [[] for i in range(0, len(words_pos))]
                    ### for i, word in enumerate(words_pos):
                    ###     supp_pos[i] = dic_pos[words_pos[i]]

                    # measuring time for each cluster
                    time_start_clustering = time.clock()
                    clusters_neg = cluster_sentences(Negative_Reviews, algorithm)
                    clusters_pos = cluster_sentences(Positive_Reviews, algorithm)
                    time_elapsed_clustering = (time.clock() - time_start_clustering)
                    results.write(str(time_elapsed_clustering)+"\t")
                    print("time : " + str(time_elapsed_clustering))

                    ### NOT used in final experiment
                    ### sort frequent items with respect to their support
                    ### sorted_inds_neg = np.flip(np.argsort(supp_neg))
                    ### sorted_inds_pos = np.flip(np.argsort(supp_pos))

                    # finding frequent words and their support in each cluster
                    clusters_neg_words = []
                    clusters_neg_supp = []

                    for cluster_num in range(len(clusters_neg)):
                        sentences = [Negative_Reviews[i] for i in clusters_neg[cluster_num]]
                        sentences = stop_word_remover(sentences)
                        [w_neg, s_neg] = find_frequent_items(sentences)

                        sort_i_n = np.flip(np.argsort(s_neg))
                        w_neg = [w_neg[i] for i in sort_i_n]
                        s_neg = [s_neg[i] for i in sort_i_n]
                        clusters_neg_words.append(w_neg)
                        clusters_neg_supp.append(s_neg)
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

                    # computing the algorithm score
                    # for positive clusters
                    temp = 0
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

                    # initialize vectors for saving important sentences
                    sentence_score_neg = [[0 for i in range(len(Negative_Reviews))] for cluster_num in clusters_neg]
                    sentence_score_pos = [[0 for i in range(len(Negative_Reviews))] for cluster_num in clusters_pos]
                    cluster_sentence_neg = [[[] for i in range(5)] for cluster_num in clusters_neg]
                    cluster_score_neg = [[0 for i in range(5)] for cluster_num in clusters_neg]
                    cluster_sentence_pos = [[[] for i in range(5)] for cluster_num in clusters_pos]
                    cluster_score_pos = [[0 for i in range(5)] for cluster_num in clusters_pos]
                    # finding the most important sentences
                    for cluster_num in range(len(clusters_neg)):
                        w_neg = clusters_neg_words[cluster_num]
                        w_sup = clusters_neg_supp[cluster_num]
                        w_neg_dic = dict(zip(w_neg, w_sup))

                        ### This part was for finding the most length of reviews to use in
                        ### review score formula. However, we did not use it for final experiment
                        ### since it didnt work well.
                        ### max_len_rev = 0
                        ### for index, sentence in enumerate(clusters_neg[cluster_num]):
                        ###     rev = Negative_Reviews[sentence]
                        ###     temp = len(rev)
                        ###     if temp > max_len_rev:
                        ###         max_len_rev = temp

                        for index, sentence in enumerate(clusters_neg[cluster_num]):
                            rev = Negative_Reviews[sentence]
                            rev = rev.split()
                            score = 0

                            for i in range(0, len(w_neg)):
                                if w_neg[i] in rev:
                                    score = score + w_neg_dic[w_neg[i]]
                            sentence_score_neg[cluster_num][sentence] = score/(len(rev)+1)
                        sorted_sentence_score_ind = np.flip(np.argsort(sentence_score_neg[cluster_num]))
                        for i in range(0, 5):
                            cluster_score_neg[cluster_num][i] = sentence_score_neg[cluster_num][sorted_sentence_score_ind[i]]
                            cluster_sentence_neg[cluster_num][i] = Negative_Reviews[sorted_sentence_score_ind[i]]

                    for cluster_num in range(len(clusters_pos)):
                        w_pos = clusters_pos_words[cluster_num]
                        w_sup = clusters_pos_supp[cluster_num]
                        max_w_sup = max(w_sup)
                        w_pos_dic = dict(zip(w_pos, w_sup))
                        max_len_rev = 0
                        ### for index, sentence in enumerate(clusters_pos[cluster_num]):
                        ###     rev = Negative_Reviews[sentence]
                        ###     temp = len(rev)
                        ###     if temp > max_len_rev:
                        ###         max_len_rev = temp
                        for index, sentence in enumerate(clusters_pos[cluster_num]):
                            rev = Positive_Reviews[sentence]
                            rev = rev.split()
                            score = 0

                            for i in range(0, len(w_pos)):
                                if w_pos[i] in rev:
                                    score = score + w_pos_dic[w_pos[i]]
                            sentence_score_pos[cluster_num][sentence] = score/(len(rev)+1)
                        sorted_sentence_score_ind = np.flip(np.argsort(sentence_score_pos[cluster_num]))
                        for i in range(0, 5):
                            cluster_score_pos[cluster_num][i] = sentence_score_pos[cluster_num][sorted_sentence_score_ind[i]]
                            cluster_sentence_pos[cluster_num][i] = Positive_Reviews[sorted_sentence_score_ind[i]]
                    # writing the results in record file
                    record_file = open(name + "_" + str(nb_clusters) + "_" + str(nb_comments) + "/" + str(hotel_count) + "_" + Hotel_name + ".txt", "w")
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
                            record_file.write("score " + str(cluster_score_neg[cluster_num][i]) + ": " + str(cluster_sentence_neg[cluster_num][i]) + "\n")
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
                            record_file.write("score " + str(cluster_score_pos[cluster_num][i]) + ": " + str(cluster_sentence_pos[cluster_num][i]) + "\n")
                            record_file.write("--------------------\n")
                    ### writing frequent items of all reviews in the files
                    ### not used in final experiments
                    # record_file.write("===================\n")
                    # record_file.write("===Frequent Items==\n")
                    # record_file.write("===================\n")
                    # record_file.write("======Negative=====\n")
                    # for i in range(0, min(10, len(words_neg))):
                    #     record_file.write(words_neg[sorted_inds_neg[i]] + " : " + str(supp_neg[sorted_inds_neg[i]]) + "\n")
                    # record_file.write("======Positive=====\n")
                    # for i in range(0, min(10, len(words_pos))):
                    #     record_file.write(words_pos[sorted_inds_pos[i]] + " : " + str(supp_pos[sorted_inds_pos[i]]) + "\n")

                    #### writing frequent pairs in the file
                    # record_file.write("===================\n")
                    # record_file.write("===Frequent Pairs==\n")
                    # record_file.write("===================\n")
                    # record_file.write("======Negative=====\n")
                    # neg_keys = list(pairs_neg_dict.keys())
                    # for i in range(0, min(10, len(neg_keys))):
                    #     record_file.write(neg_keys[i] + " : " + "\t")
                    #     adjs = pairs_neg_dict[neg_keys[i]]
                    #     for j in range(0, len(adjs)):
                    #         record_file.write(adjs[j] + "\t")
                    #     record_file.write("\n")
                    #     record_file.write("--------------------\n")
                    # record_file.write("======Positive=====\n")
                    # pos_keys = list(pairs_pos_dict.keys())
                    # for i in range(0, min(10, len(pos_keys))):
                    #     record_file.write(pos_keys[i] + " : " + "\t")
                    #     adjs = pairs_pos_dict[pos_keys[i]]
                    #     for j in range(0, len(adjs)):
                    #         record_file.write(adjs[j] + "\t")
                    #     record_file.write("\n")
                    #     record_file.write("--------------------\n")

                    record_file.write("=====Summary=====\n")
                    record_file.write("=====Positive====\n")
                    for cluster_num in range(len(clusters_pos)):
                        for i in range(0, 1):
                            record_file.write(str(cluster_num) + ": " + str(cluster_sentence_pos[cluster_num][i]) + "\n")
                    record_file.write("=====Negative====\n")
                    for cluster_num in range(len(clusters_neg)):
                        for i in range(0, 1):
                            record_file.write(str(cluster_num) + ": " + str(cluster_sentence_neg[cluster_num][i]) + "\n")

                    hotel_count = hotel_count + 1
                    print(name)
                    print(Hotel_name)
                    time_elapsed = (time.clock() - time_start)
                    print("time : " + str(time_elapsed))

                    algorithm_file.write("Number of reviews : " + str(Number_of_Reviews) + " for Hotel : " + Hotel_name + "\n")
                    algorithm_file.write("time : " + str(time_elapsed) + "\n")
                    algorithm_file.write("number of common words between clusters : " + str(common_words_count) + "\n")
                    results.write(str(common_words_count/(nb_clusters*(nb_clusters-1)*5)) + "\n")