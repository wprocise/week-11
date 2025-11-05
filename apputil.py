import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

def kmeans(X, k):

    model = KMeans(n_clusters=k, random_state=42)
    model.fit(X)

