"""
Assignment 1: K-Means Clustering and Fuzzy C Means Clustering

Diana Johnson
02-17-2026
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def initialize_centroid(data):
    """
    initialize_centroid() : For each feature find out the minimum and maximum values. Then randomly
            select a value for each feature (using uniform distribution) from the min-max range to 
            initialize one complete centroid.

    Parameters:
        data: pd dataframe, dataset of features for analysis

    Returns:
        centroid: center for given data
    """

    # num of columns in data is the num of features
    num_features = data.shape[1]
    # make empty array
    centroid = np.zeros(num_features)

    # Initialize random value for each dimension (feature)
    for i in range(num_features):
        f_min = data.iloc[:, i].min()
        f_max = data.iloc[:, i].max()
        centroid[i] = np.random.uniform(f_min, f_max)

    # return centroid
    return centroid

def assign_opt_clusters(data, k, centroids, max_iters, epsilon):
    """
    assign_opt_clusters() : Calculates the best centroids for a given set of data and given 
                        k value

        Parameters:
        data: pd dataframe, dataset of features for analysis
        k: int, number of clusters
        centroids: list, centers for clusters
        max_iters: maximum iterations for finding opt centers
        tolerance: target error/difference between iterations

    Returns:
        centroids: optimal centers per cluster
        centroid_labels: best center for each data point
    """

    X = data.values
    centroids = np.array(centroids)

    # assign each point to a cluster
    for _ in range(max_iters):
        centroid_labels = np.zeros(len(X))

        # assign points to centroids
        for i in range(len(X)):
            dists = []
            for j in range(k):
                # calculate euclidean dist to every cluster
                d = np.sqrt(np.sum((X[i] - centroids[j]) ** 2))
                dists.append(d)
            
            # take min dist location as cluster assignment
            centroid_labels[i] = np.argmin(dists)
        
        # update centers, and initialize using current center shape
        new_centroids = np.zeros_like(centroids)

        # Re-compute centroids
        for j in range(k):
            # pull out each point that matches the label for the current cluster j
            cluster_points = X[centroid_labels == j]

            if len(cluster_points) > 0:
                # get a mean for each feature
                new_centroids[j] = np.mean(cluster_points, axis=0)
            else:
                # keep old center
                new_centroids[j] = centroids[j]
        
        # Check for convergence
        # linalg.norm calculates euclidean dist
        if np.linalg.norm(new_centroids - centroids) < epsilon:
            break
            
        centroids = new_centroids
    
    return centroids, centroid_labels

def plot_3d(data, labels, centroids, title):
    X = data.values

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(X[:,0],X[:,1],X[:,2], c=labels)
    ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c='blue',marker='o')

    ax.set_xlabel("Age")
    ax.set_ylabel("Impulsiveness (BIS-11)")
    ax.set_zlabel("Sensation Seeking (ImpSS)")

    ax.set_title(title)

    plt.show()

def main():
    # Load drug_consumption dataset
    data = pd.read_csv("drug_consumption.data", header=None)

    """
    Assignment Part 1
    """

    # PART A
    #=============================================================
    
    # Pull out all rows of required features (columns)
    # Age (1), Impulsiveness (11), SS (12)
    features = data.iloc[:,[1,11,12]]
    
    # 2 clusters
    k = 2

    # Initialize centers
    initial_centroids = []
    for _ in range(k):
        initial_centroids.append(initialize_centroid(features))
    
    print(initial_centroids)
    
    centroids, labels = assign_opt_clusters(features, k, initial_centroids, max_iters=200, epsilon=1e-6)

    # plot results
    plot_3d(features, labels, centroids, "K-means: k=2")



    # PART B
    #=============================================================



    """
    Assignment Part 2
    """



main()