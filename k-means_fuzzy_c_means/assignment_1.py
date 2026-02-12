"""
Assignment 1: K-Means Clustering and Fuzzy C Means Clustering

Diana Johnson
02-17-2026
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Load drug_consumption dataset
data = pd.read_csv("drug_consumption.data", header=None)

# Pull out all rows of required features (columns)
# Age (1), Impulsiveness (11), SS (12)
features = data.iloc[:,[1,11,12]]

def initialize_centroid(data):
    """
    Randomly initialize one centroid by finding the min and max value of each feature
    and applying it to a random uniform distribution 

    Parameters:
        data: the dataset

    Returns:
        centroid: random center list
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
