import json
import bson
import csv
from bson.objectid import ObjectId



from pymongo import MongoClient
from flask import g

from . import app

MAX = 500

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

def connect_db():

    """Connects to the specific database."""
    client = MongoClient('mongodb://localhost:27017/')
    db = client['test-database']
    
    return db


def get_db():

    """Opens a new database connection if there is none yet for the

    current application context.

    """

    if not hasattr(g, 'mongodb'):
        g.mongodb = connect_db()
        return g.mongodb


#@app.teardown_appcontext
def close_db(error):

    """Closes the database again at the end of the request."""

    if hasattr(g, 'mongodb'):
        g.mongodb.close()

def get_colletions():
    db = get_db()

    return db.collection_names()
        
def csv_handler(file_name,colletion_name):
    data_set = []
    db = get_db()
    collection = db[colletion_name]
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

#def run_mds(colletion_name):
    import numpy as np
    from sklearn import manifold
    from sklearn.metrics import euclidean_distances
    from sklearn.decomposition import PCA
    #n_samples = 20
    seed = np.random.RandomState(seed=3)
    #X_true = seed.randint(0, 20, 2 * n_samples).astype(np.float)
    #print X_true
#    data_set = colletion.
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

 
class CustomEncoder(json.JSONEncoder):
    """A C{json.JSONEncoder} subclass to encode documents that have fields of
    type C{bson.objectid.ObjectId}, C{datetime.datetime}
    """
    def default(self, obj):
        if isinstance(obj, bson.objectid.ObjectId):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return json.JSONEncoder.default(self, obj)
            


    
def case_query(collection_name):
    db = get_db()
    collection = db[collection_name]
    ec = CustomEncoder()
    return ec.encode(list(collection.find()))
    