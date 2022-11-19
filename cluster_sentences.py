import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from aux_functions import word_tokenizer


def cluster_sentences(sentences, clust_algorithm):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=word_tokenizer,
                                       stop_words=stopwords.words('english'),
                                       max_df=0.9,
                                       min_df=0.1,
                                       lowercase=True)

    # builds a tf-idf matrix for the sentences
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    clust_algorithm.fit(tfidf_matrix)
    clusters = collections.defaultdict(list)
    for i, label in enumerate(clust_algorithm.labels_):
        clusters[label].append(i)
    return dict(clusters)
