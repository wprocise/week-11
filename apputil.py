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



