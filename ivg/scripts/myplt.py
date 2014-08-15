# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# Licence: BSD

print(__doc__)
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
import csv

import sys

file_name = sys.argv[1]
X_true = []
t_data = open(file_name)
for data in t_data:
    X_true.append(list(data[:-1].split(',')))
#X_true.reverse()
clf = PCA(n_components=2)
X_true = clf.fit_transform(X_true)


fig = plt.figure(1)
ax = plt.axes([0., 0., 1., 1.])

plt.scatter(X_true[:, 0], X_true[:, 1], c='r', s=20)

plt.legend(('True position', 'MDS', 'NMDS'), loc='best')
plt.show()


