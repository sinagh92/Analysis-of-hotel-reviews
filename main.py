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
from aux_functions import find_frequent_items,stop_word_remover,split_uppercase,get_words_for_each_cluster,save_important_sentences,write_results,compute_algorithm_score
from Modules import Reviews
from cluster_sentences import cluster_sentences
from Frequent_Pairs import Frequent_Pairs

# Uncomment if required
# warnings.filterwarnings("ignore")
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')


# Dataset information
# https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe




def main(separate_sentences_analysis,nb_comments,compute_freq_items,compute_freq_pairs):
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
    
    if not os.path.exists('results'):
        os.makedirs('results')
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
        log = open("results\clusters_" + str(nb_clusters) + "_" + str(nb_comments) + ".tsv", "w")

        for name, algorithm in clustering_algorithms:
            if not os.path.exists("results/"+name + "_" + str(nb_clusters) + "_" + str(nb_comments)):
                os.mkdir("results/"+name + "_" + str(nb_clusters) + "_" + str(nb_comments))
                print("Directory ", name, " Created ")
            else:
                print("Directory ", name, " already exists")
            # write the name of algorithm on top of data in record file
            log.write("Hotel Name" + "\t" + "reviews" + "\t" + "time" + "\t" + name + "\n")

            hotel_count = 0

            for Hotel_name in unique_hotels:
                time_start = time.clock()
                
                Pos_reviews = Reviews() 
                Neg_reviews = Reviews() 
                
                
                
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
                    Negative_Reviews = Negative_Reviews[1:100]
                    Positive_Reviews = Positive_Reviews[1:100]
                    Neg_reviews.all_reviews = Negative_Reviews
                    Pos_reviews.all_reviews = Positive_Reviews
                    
                    ### separating each sentence in each review
                    if separate_sentences_analysis == True:
                    ### this part is not used in final experiments
                        for review in Negative_Reviews:
                            sentences = split_uppercase(review).split('.')
                            all_reviews_neg.append(review.lower())
                            for sentence in sentences:
                                if len(sentence) < 10:
                                    sentences.remove(sentence)
                                elif sentence != "Negative":
                                    all_sentences_neg.append(sentence.lower())
                        Neg_reviews.all_sentences_neg = all_sentences_neg
                        
                        for review in Positive_Reviews:
                            sentences = split_uppercase(review).split('.')
                            all_reviews_pos.append(review.lower())
                            for sentence in sentences:
                                if len(sentence) < 10:
                                    sentences.remove(sentence)
                                elif sentence != "Positive":
                                    all_sentences_pos.append(sentence.lower())
                        Pos_reviews.all_sentences_neg = all_sentences_neg

                    print("Number of reviews : " + str(Number_of_Reviews) + " for Hotel : " + Hotel_name)
                    log.write(Hotel_name + "\t" + str(Number_of_Reviews) + "\t")



                    # measuring time for each cluster
                    time_start_clustering = time.clock()
                    Neg_reviews.clusters = cluster_sentences(Negative_Reviews, algorithm)
                    Pos_reviews.clusters = cluster_sentences(Positive_Reviews, algorithm)
                    time_elapsed_clustering = (time.clock() - time_start_clustering)
                    log.write(str(time_elapsed_clustering)+"\t")
                    print("time : " + str(time_elapsed_clustering))



                    # finding frequent words and their support in each cluster
                    get_words_for_each_cluster(Neg_reviews)   

                    get_words_for_each_cluster(Pos_reviews)   
                    
                    # computing the algorithm score
                    common_words_count = compute_algorithm_score(Pos_reviews,Neg_reviews)
                    
                    save_important_sentences(Neg_reviews, Pos_reviews)
                    
                    # writing the results in record file
                    write_results(Pos_reviews,Neg_reviews, name, nb_clusters,nb_comments, hotel_count,Hotel_name,compute_freq_items,compute_freq_pairs)
                    
                    


                    hotel_count = hotel_count + 1
                    print(name)
                    print(Hotel_name)
                    time_elapsed = (time.clock() - time_start)
                    print("time : " + str(time_elapsed))
                    
                    # open another file for each algorithm to write more detailed results
                    algorithm_file = open("results/"+name + "_" + str(nb_clusters) + "_" + str(nb_comments) + "/" + "report.txt", "w")
                    algorithm_file.write("Number of reviews : " + str(Number_of_Reviews) + " for Hotel : " + Hotel_name + "\n")
                    algorithm_file.write("time : " + str(time_elapsed) + "\n")
                    algorithm_file.write("number of common words between clusters : " + str(common_words_count) + "\n")
                    log.write(str(common_words_count/(nb_clusters*(nb_clusters-1)*5)) + "\n")

if __name__ == "__main__":
    
    # Analysis options:
    separate_sentences_analysis = False
    compute_freq_items = False
    compute_freq_pairs = False
    
    # we use hotel data which have more than nb_comments reviews
    nb_comments = 5000
    
    main(separate_sentences_analysis,nb_comments,compute_freq_items,compute_freq_pairs)
        
    