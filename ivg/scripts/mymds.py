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
seed = np.random.RandomState(seed=3)
my_data=[]
data = open(file_name)
reader = csv.reader(data)
for line in reader:
    my_data.append([int(c) for c in line])
data.close()


#  start small dont take all the data, 
#  its about 200k records
subset = my_data[:1000]
similarities = euclidean_distances(subset)

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
                   dissimilarity='precomputed',random_state=seed,n_jobs=-1)

pos = mds.fit(similarities).embedding_

nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                     dissimilarity='precomputed',random_state=seed, n_jobs=-1,
                    n_init=1)
npos = nmds.fit_transform(similarities, init=pos)



clf = PCA(n_components=2)
X_true = clf.fit_transform(my_data)
pos = clf.fit_transform(pos)
npos = clf.fit_transform(npos)

out3 = open('nposdata','w')
for x in npos:
    out3.write(str(x[0])+','+str(x[1])+'\n')

#fig = plt.figure(1)
#ax = plt.axes([0., 0., 1., 1.])

#plt.scatter(X_true[:, 0], X_true[:, 1], c='b', s=20)
#plt.scatter(pos[:, 0], pos[:, 1], c='r', s=20)
#plt.scatter(npos[:, 0], npos[:, 1], s=20, c='y')
#plt.legend(('True position', 'MDS', 'NMDS'), loc='best')


#for label, x, y in zip(xrange(150), npos[:, 0], npos[:, 1]):
#    plt.annotate(
#        label,
#        (x,y),
#        )

#plt.show()
