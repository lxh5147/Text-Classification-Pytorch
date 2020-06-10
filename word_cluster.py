from __future__ import division

import operator

import numpy
import spacy
from sklearn.cluster import KMeans

# find similar words example, use more advanced models in the pipeline

# step 1: download a model: python -m spacy download en_core_web_lg

# step 2: load the model with a nlp pipeline properly configured, refer to: https://spacy.io/usage/processing-pipelines

nlp = spacy.load("en_core_web_lg", disable=["tagger", "parser"])

# step 3: apply the pipeline to micro text to get a document object
tokens = nlp("dog cat banana kkdfd")

# load the text and parse
texts = ["dog cat banana.", "fight a fly."]


def parse_texts_to_tokens(texts, top_k=1000):
    tokens = {}
    counts = {}
    for text in texts:
        doc = nlp(text)
        for token in doc:
            if not token.is_stop and not token.is_punct:
                if token.text not in tokens:
                    tokens[token.text] = token
                    counts[token.text] = 1
                else:
                    counts[token.text] += 1
    # order by freq and pick the top k most frequent tokens
    sorted_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)
    return [tokens[text] for text, _ in sorted_counts[:top_k]]


tokens = parse_texts_to_tokens(texts)


# cluster tokens

def build_word_vector_matrix(tokens):
    '''Return the vectors and labels for the first n_words in vector file'''
    numpy_arrays = []
    labels_array = []

    for token in tokens:
        labels_array.append(token.text)
        numpy_arrays.append(token.vector)

    return numpy.array(numpy_arrays), labels_array


df, labels_array = build_word_vector_matrix(tokens)


# TODO: make it more efficient
def build_word_clusters(df, n_clusters=10):
    kmeans_model = KMeans(n_clusters=n_clusters)
    kmeans_model.fit(df)
    return kmeans_model.labels_


cluster_labels = build_word_clusters(df, n_clusters=2)


def find_word_clusters(labels_array, cluster_labels):
    cluster_to_words = {}
    # cluster labels contain the cluster id for each item
    for i, c in enumerate(cluster_labels):
        if c in cluster_to_words:
            cluster_to_words[c].append(labels_array[i])
        else:
            cluster_to_words[c] = [labels_array[i]]
    return cluster_to_words


word_clusters = find_word_clusters(labels_array, cluster_labels)

# show word clusters
for c in word_clusters:
    print('cluster id {} contains the following words: {}'.format(c, word_clusters[c]))
