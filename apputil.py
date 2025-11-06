import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

"""Define kmeans and perform clustering on numerical numpy array.
    Returns tuples, those being centroids and labels.
"""
def kmeans(X, k):

    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)
    
    centroids = model.cluster_centers_
    labels = model.labels_

    return (centroids, labels)

"""Load 'diamonds' dataset from seaborn library.
    Keep numeric columns only
"""
diamonds = sns.load_dataset("diamonds")
numeric_diamonds = diamonds.select_dtypes(include=[np.number])

def kmeans_diamonds(n, k):

    # Take first n rows and convert to NumPy array
    X = numeric_diamonds.iloc[:n].to_numpy()

    # Call the previously defined kmeans function
    return kmeans(X, k)










