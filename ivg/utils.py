import json
import bson
import csv
import math
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
    return ec.encode(list(collection.find({}, {'coordinate' : 0 })))


def kmeans_query(collection_name, selected_points, algorithm):
    db = get_db()
    collection = db[collection_name]
    points = list(collection.find({}, {'_id':0, algorithm:1}))
    centroids = list(collection.find({'_id':{'$in':selected_points}}, {'_id':0, algorithm:1}))
    return Kmeans(points, centroids)
    



def kmeans_2_query(collection_name):
    db = get_db()
    collection = db[collection_name]
    ec = CustomEncoder()
    return ec.encode(list(collection.find({}, {'coordinate' : 0 })))



#kmeans part starts
    
#watch out: float flaw 
def squared_euclidean_distance(pointA, pointB):
    if len(pointA) != len(pointB):
        raise ValueError('the two points should have the same degree of dimension')

    dimension = len(pointA)
    distance = 0.0
    for i in range(dimension):
        distance += math.pow(pointA[i] - pointB[i], 2)

    return distance
    
    
def assignment(points, centroids):
    '''
    assgin points to k clusters, the first step of k-means iteration
    Randomly pick k points as centroids then assgin points to the nearest centroid
    '''
    k = len(centroids)
    num_of_points = len(points)
    cluster_matrix = [[0 for i in range(num_of_points)] for j in range(k)]

    for i in range(len(points)):
        distance = float('inf')
        assign_to = None
        for j in range(len(centroids)):
            e_distance = squared_euclidean_distance(points[i], centroids[j])
            if e_distance <= distance:
                distance = e_distance
                assign_to = j
        cluster_matrix[assign_to][i] = 1
        
    return cluster_matrix


def update_centroids(points, cluster_matrix):
    new_centroids = []
    dimension = len(points[0])
    for cluster in cluster_matrix:
        coordinate = [0 for i in range(dimension)]
        for position in range(len(cluster)):
            if cluster[position] == 1:
               coordinate = map(sum, zip(coordinate, points[position]))
        count = float(cluster.count(1))
        mean = map(lambda x:x/count, coordinate)
        new_centroids.append(mean)    
    return new_centroids

def KMeans(points,centroids):
    #import pprint
    #centroids = random.sample(points, k)
    cluster_matrix = assignment(points, centroids)
    #cluster_matrix
    centroids = update_centroids(points, cluster_matrix)
    return cluster_matrix,centroids
    #plot(points, cluster_matrix, centroids)
    