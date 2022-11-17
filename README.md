# CSC578D_data_mining_project
Analyzing Hotel Reviews and Summarizing them based on Frequent Words
This project is based on 616K Hotel Reviews Data in Europe from Booking.com provided in the following link:

[Dataset for this project from Kaggle website](https://www.kaggle.com/datasets/jiashenliu/515k-hotel-reviews-data-in-europe)

## Introduction

There are more than thousands of hotel reviews in popular hotel booking websites. These reviews are beneficial in two ways. First, managers of the hotels can use these reviews to improve the quality of their services. Second, future customers can read these reviews and select their hotels based on their expectations. However, reading this amount of reviews is very time consuming and unreasonable. Therefore, it is important to find techniques to summarize the reviews into important points so that people can get more information in less time and also hotel managers find the most important problems of their hotels faster.

In this project, we work on one of Kaggle datasets that is introduced in the next section. We use a clustering algorithm to cluster positive and negative reviews into different categories then by finding frequent words in each cluster, we choose the most important reviews and summarize each cluster by choosing an appropriate sentence for that cluster.

## Dataset
Our dataset contains 515k hotel reviews of 1492 hotels around Europe. These reviews are categorized into positive and negative reviews for each hotel. There are some other data such as reviewer’s nationality, score which each user has given to the hotel, and hotel’s latitude and altitude which are not used in our project. Since we want to benefit from largeness of data, we only run our algorithm on hotels which have more than 5000 reviews which are 31 hotels in this dataset.

## Our Approach
Our main idea in this project is to perform clustering on hotel reviews. In order to do so, first, we need to pre-process the data. 

### Step 1: Preprocessing: 
In this step, each review is tokenized into words. Then, stop words are removed from these tokens. Stop words are the words that are common in a language but does not have any useful information in natural language processing. Some examples of stop words are “the”, “a”, “and”, “an”, “in”. After that, we use Term-Frequency and Inverse-Document-Frequency as feature extraction method for each review. 

### Step 2: Clustering
In this step, we perform a clustering algorithm (BIRCH or K-means) on positive and negative reviews separately. The result is N clusters for positive reviews and N clusters for Negative reviews of a hotel. (N could be between 5 to 15, and is determined later) Now, we find the frequent words in sentences of each cluster separately to find out which words are more frequently used by the costumers.

### Step 3: Giving score to each review
In this step, we give scores to each review based on the frequent words which is present in that review. In order to do so, first we define word scores. We use the support of a frequent word as its score. 

〖Score〗_i=Support(〖Frequent_word〗_i )

![](https://myoctocat.com/assets/images/base-octocat.svg)


Then, we define a vector M of all frequent words in each cluster. For each review, we have a M’ vector which has value 1 if that frequent word is present in the review and value 0 if that word is not present in the review.

M^'=[0 0 1 0…0 1  0 ]        M=[Room   Staff   Restaurant  …Location ]

Therefore, for each review, the sum of supports for each frequent word is calculated. Since some reviews may have a lot of words and are not summarized enough, we try to weight our algorithm to prefer smaller sentences which contain most frequent words. We have seen that smaller sentences are more informative. Therefore, we use the following formula for review scores.

Score of a Review=  (∑_iϵ M^ ▒S_i )/N

Where N is the number of words in that review. 

### Step 4: choosing best reviews
In this step, we choose 5 most scored sentences for each cluster. We can also use the first most scored sentence as the representative of that cluster.

## Other efforts
We tried preprocessing the data by finding the lemma of each word. In order to do so, we used wordlemmatizer (NLTK library). It seemed promising at first since we assumed it would give us the root of each word. But since it had a lot of errors, for example “shower” was converted to “show” or “location” was converted to “locat”, it did not work well on our problem.
  
We also tried to find frequent pairs for each hotel. We did it in two ways. First trying to find frequent pairs generally from all reviews (positives or negatives separately). The second method was to find Nouns and Adjectives of each review, and trying to find pairs that have one noun and one adjective. However, these two methods were not useful since there were a lot of meaningless pairs in the data.
  
The next method that we tried was to extract sentences from reviews and use them as basic elements of our dataset. Since the person who uploaded the dataset had removed all punctuations from the dataset, it was hard to do so. So we tried to find sentences by finding the words that start with capital letter and followed by lowercase letters. We also considered some words such as “I”, “TV” “AC” as exceptions. Then we did all of the above using sentences instead of reviews. Since we did not have punctuations and many people did not write meaningful sentences or they just used words only, this idea did not improve our clustering as well. So we decided to work with reviews themselves.
  
  
##  Conclusion and future work
Overall, we found our method very useful since finding the important topics in reviews is very hard. By using this method, hotel managers and future visitors can find the important topics from the reviews much faster. We learned how to implement clustering algorithms in Python and how to use different libraries for Natural Language Processing. We also learned the difference between BIRCH and K-means in practice. It was very interesting for us to develop a data mining approach, work with big data and get meaningful results. 

If we had more time, we would like to test other clustering algorithms and other datasets. One of the challenges in this problem is choosing the right number of clusters. Therefore we would like to try algorithms that predict number of clusters before using K-means and BIRCH.


