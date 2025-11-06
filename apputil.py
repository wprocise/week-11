import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

"""Define kmeans and perform clustering on numerical numpy array.
    Returns tuples, those being centroids and labels.
    Perform k-means clustering on numerical numpy array.
"""
def kmeans(X, k):

    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    
    centroids = model.cluster_centers_
    labels = model.labels_

    return (centroids, labels)

"""Load 'diamonds' dataset from seaborn library.
    Keep numeric columns only, and save them in a dataframe.
    Run k-means on first n rows of numeric diamonds data.
"""
diamonds = sns.load_dataset("diamonds")
numeric_diamonds = diamonds.select_dtypes(include=[np.number])

def kmeans_diamonds(n, k):

    # Take first n rows and convert to NumPy array
    X = numeric_diamonds.iloc[:n].to_numpy()

    # Call the previously defined kmeans function
    return kmeans(X, k)

def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds(n, k) exactly n_iter times and
    return the average runtime in seconds.
    n = # of rows from dataset, k = # of clusters, n_iter = # of iterations.
    """
    times = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds (n, k)
        elapsed = time() - start
        times.append(elapsed)

    return sum(times) / len(times)











