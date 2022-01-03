# import statements
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt  # create blobs
data = make_blobs(n_samples=200, n_features=2, centers=4,
                  cluster_std=1.6, random_state=50)  # create np array for data points
points = data[0]  # create scatter plot
plt.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='viridis')
plt.xlim(-15, 15)
plt.ylim(-15, 15)
