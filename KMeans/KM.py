from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
np.random.seed(11)

from sklearn.cluster import KMeans

##################################################################

means = [[2, 2], [8, 8], [5, 6]]
cov = [[3, 0], [0, 1]]
N = 500
X0 = np.random.multivariate_normal(means[0], cov, N)
X1 = np.random.multivariate_normal(means[1], cov, N)
X2 = np.random.multivariate_normal(means[2], cov, N)

X = np.concatenate((X0, X1, X2), axis = 0)
K = 3

original_label = np.asarray([0]*N + [1]*N + [2]*N).T


kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

##################################################################

pred_label = kmeans.predict(X)

center = kmeans.cluster_centers_

##################################################################

"""
    TODO:
        Visualize sample X colored by the corresponding label in pred_label
"""