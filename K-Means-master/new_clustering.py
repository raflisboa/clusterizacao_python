import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import sklearn
from sklearn import cluster, datasets

import warnings
import sys

print("Python Version : ",sys.version)
print("Scikit-Learn Version : ",sklearn.__version__)

warnings.filterwarnings('ignore') ## We'll silent future warnings using this command.
np.set_printoptions(precision=3)

## Beow magic function fits plot inside of current notebook. 
## There is another option to it (%matplotlib notebook) which opens plot in new notebook.
# %matplotlib inline

# We'll create a dataset with 250 samples, 2 features and 5 cluster centers using scikit-learn's make_blobs method.

samples, clusters = datasets.make_blobs(n_samples=250, n_features=2, centers=5, cluster_std=0.7, random_state=12345)
print('Dataset size : ', samples.shape, clusters.shape)
print('Cluster names : ',set(clusters))