# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# Licence: BSD
import csv
import sys

import numpy as np


from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

file_name = sys.argv[1]

my_data=[]
csv_data = open(file_name)
reader = csv.reader(csv_data)
for line in reader:
    my_data.append([int(c) for c in line])
csv_data.close()



#n_samples = 20
seed = np.random.RandomState(seed=3)
#X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
#print X_true
print len(my_data)
X_true = my_data[:10]
#X_true = X_true.reshape((n_samples, 2))
#print X_true
# Center the data
#X_true -= X_true.mean()

similarities = euclidean_distances(X_true)

# Add noise to the similarities
#noise = np.random.rand(n_samples, n_samples)
#noise = noise + noise.T
#noise[np.arange(noise.shape[0]), np.arange(noise.shape[0])] = 0
#similarities += noise

mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, random_state=seed,
                   dissimilarity="precomputed", n_jobs=-1)
pos = mds.fit(similarities).embedding_


nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
                    dissimilarity="precomputed", random_state=seed, n_jobs=-1,
                    n_init=1)
npos = nmds.fit_transform(similarities, init=pos)
# Rescale the data
#pos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((pos ** 2).sum())
#npos *= np.sqrt((X_true ** 2).sum()) / np.sqrt((npos ** 2).sum())

# Rotate the data
clf = PCA(n_components=2)
X_true = clf.fit_transform(X_true)

pos = clf.fit_transform(pos)

npos = clf.fit_transform(npos)

in_c = open('data_count_by_region.1000')
out = open('rtmp_data','w')
out1 = open('rtmp_data1','w')
out2 = open('rtmp_data2','w')
for x in X_true:
    out.write(str(x[0])+','+str(x[1])+'\n')
out.close()
for x in pos:
    out1.write(str(x[0])+','+str(x[1])+'\n')
out.close()
for x in npos:
    out2.write(str(x[0])+','+str(x[1])+'\n')
out.close()

#fig = plt.figure(1)
#ax = plt.axes([0., 0., 1., 1.])
print X_true
print pos
print npos
#plt.scatter(X_true[:, 0], X_true[:, 1], c='r', s=20)
#plt.scatter(pos[:, 0], pos[:, 1], s=20, c='g')
#plt.scatter(npos[:, 0], npos[:, 1], s=20, c='b')
#plt.legend(('True position', 'MDS', 'NMDS'), loc='best')

#similarities = similarities.max() / similarities * 100
#similarities[np.isinf(similarities)] = 0

# Plot the edges
#start_idx, end_idx = np.where(pos)
#a sequence of (*line0*, *line1*, *line2*), where::
#            linen = (x0, y0), (x1, y1), ... (xm, ym)
#segments = [[X_true[i, :], X_true[j, :]]
#            for i in range(len(pos)) for j in range(len(pos))]
#values = np.abs(similarities)
#lc = LineCollection(segments,
#                    zorder=0, cmap=plt.cm.hot_r,
#                    norm=plt.Normalize(0, values.max()))
#lc.set_array(similarities.flatten())
#lc.set_linewidths(0.5 * np.ones(len(segments)))
#ax.add_collection(lc)

#plt.show()

