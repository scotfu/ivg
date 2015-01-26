import csv

from pymongo import MongoClient
import numpy as np
from celery import Celery, task

def connect_db():

    """Connects to the specific database."""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['test-database']
    
    return db

db = connect_db()

MAX = 500

app = Celery('ivg.tasks', broker='amqp://guest@localhost//')


@app.task
def add(a, b):
    return a + b 


@app.task(name='ivg.tasks.csv_handler')    
def csv_handler(file_name,collection_name, url, content):
    data_set = []
    collection = db[collection_name]
    case_collection = db['fsc_case']
    numeric_headers = []
    with open(file_name) as file_handler:
        reader = csv.reader(file_handler)
        header = reader.next()
        n = len(header)
        count = 1
        for row in reader:
            data = {'coordinate':[]}
            for column in range(n):
                try:
                    row[column] = float(row[column])
                    data['coordinate'].append(row[column])
                    if count == 1:
                        numeric_headers.append(header[column])
                except ValueError:
                    data[header[column]] = row[column]
            data_set.append(data['coordinate'])        
            try:
                data['_id'] = count
                collection.insert(data)
            except Exception,e:
                print e
            if count >= MAX:
                break
            count += 1
                
#   run_mds(file_name)           


    from sklearn import manifold
    from sklearn.metrics import euclidean_distances
    from sklearn.decomposition import PCA
    #n_samples = 20
    seed = np.random.RandomState(seed=3)
    #X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    #print X_true

    X_true = data_set[:MAX]
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
    for n in range(len(npos)):
        collection.update({'_id' : n+1 },
                          {'$set':
                           {'mds' : list(pos[n]),
                            'nmds' : list(npos[n]),
                            'pca' : list(X_true[n]), 
                           }
                          })
    print url,content
    case_collection.insert({'name':collection_name, 'header':numeric_headers, 'url':url,
                            'content': content})

